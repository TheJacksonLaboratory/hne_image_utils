import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon
import numpy as np
import shapely
import geojson
import openslide
from openslide import open_slide
# from openslide.deepzoom import DeepZoomGenerator
import matplotlib.pyplot as plt
from shapely.geometry import shape, GeometryCollection
from shapely.strtree import STRtree
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.affinity import scale
from shapely.affinity import translate
from shapely.ops import unary_union
import pandas as pd

def mask_to_polygons_layer(mask, scale_x = 1.0, scale_y = 1.0):
    """Transform each connected non-zero entry in mask into a shapely shape

    Parameter:
        mask : numpy.array
           The image mask.

        scale_x : float, optional
           Factor by which to scale mask to be of the same x dimension as the original image.
           
        scale_y : float, optional
           Factor by which to scale mask to be of the same y dimension as the original image.
           
    Returns:
        shapely.geometry.MultiPolygon: all connected shapely shapes
    """

    # This function taken from from https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    # with minor changes for upscaling

    
    all_polygons = []
    # rasterio.Affine(a, b, c, d, e, f) transforms x, y to x', y' via:
    # | x' |   | a  b  c | | x |
    # | y' | = | d  e  f | | y |
    # | 1  |   | 0  0  1 | | 1 |
    for shape, value in rasterio.features.shapes(mask.astype(np.int16), mask=(mask >0), transform=rasterio.Affine(scale_x, 0, 0, 0, scale_y, 0)):
        # commented this out from the original code. leaving it in would return just the first
        # component / polygon amongst the multiple in a disconnected mask.
        # return shapely.geometry.shape(shape)
        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons

def upscale_mask_to_polygon(image_wsi_file, tissue_mask_file):
    """Read in a tissue mask and convert it to a shapely (Multi)Polygon of the same scale as the original image.

    Parameter:
        image_wsi_file : str
           The image file to read. File should be compatiable with openslide.

        tissue_mask_file : str
           The image mask file to read.
           
    Returns:
        shapely.geometry.(Multi)Polygon: a single (Multi)Polygon representing the (non-zero entries in the) mask
    """
    
    # open the image and get its dimensions
    slide = openslide.open_slide(image_wsi_file)
    xsz, ysz = slide.dimensions
    
    # open the mask and determine its potential downscaling relative to the full image
    mask = plt.imread(tissue_mask_file)
    scale_x = xsz / mask.shape[1]
    scale_y = ysz / mask.shape[0]
    
    # confirm that the downscaling is an integer (modulo some rounding)
    tol = 10**-1
    if abs(scale_x - round(scale_x)) > tol:
        raise Exception(str(scale_x) + " is not (close to) an integer")
    if abs(scale_y - round(scale_y)) > tol:
        raise Exception(str(scale_y) + " is not (close to) an integer")
    # scale_x = int(scale_x)
    # scale_y = int(scale_y)
    if round(scale_x) != round(scale_y):
        raise Exception("scale_x (" + str(scale_x) + ") != scale_y (" + str(scale_y) + ")")

    # translate the tissue mask to an (upscaled) shapely Polygon
    print('Upscaling tissue mask by ' + str(scale_x) + ' in x and ' + str(scale_y) + ' in y')    
    tissue_polygons = mask_to_polygons_layer(mask, scale_x, scale_y)
    
    # combine any disjoint pieces in the mask to a single (Multi)Polygon
    tissue_polygon = unary_union(tissue_polygons)
    
    return tissue_polygon

def create_annotation_polygons(annotation_geojson_file):
    """Create a shapely (Multi)Polygon for each annotation class (e.g., tumor, stroma, or necrosis) in QuPath-derived GeoJSON file

    Parameter:
        annotation_geojson_file : str
           A GeoJSON file of annotations (e.g., as output by QuPath). 
           The annotations are assumed to have named classifications (e.g., tumor, stroma, necrosis).

    Returns:
        dict (str: shapely.geometry.(Multi)Polygon) : a dictionary from annotation class name to shapely (Multi)Polygon representing that annotation class.
    """
    
    # read in the annotations
    allobjects = geojson.load(open(annotation_geojson_file))
    feats = [a for a in allobjects["features"] if 'classification' in a['properties'].keys() ]
    
    # separate the annotations by class / name (e.g., tumor, stroma, etc.)
    feature_classes = [feat['properties']['classification']['name'] for feat in feats]
    feat_dict = {}
    for feat in feature_classes:
        feat_dict[feat] = []
    for feat in feats:
        feat_dict[feat['properties']['classification']['name']].append(feat)
    
    # translate the annotations to shapely Polygons, separated by annotation class
    shape_dict = {}
    for k in feat_dict.keys():
        shape_dict[k] = []
        for feat in feat_dict[k]:
            shape_dict[k].append(shape(feat["geometry"]))
    
    # collapse all Polygons of a given class into one MultiPolygon
    anno_polygons = {}
    for k in shape_dict.keys():
        anno_polygons[k] = unary_union(shape_dict[k])
    
    return anno_polygons   


def intersect_annotation_polygons_with_mask(image_wsi_file, tissue_mask_file, anno_polygons):
    """Intersect shapely (Multi)Polygons, each representing an annotation class (e.g., tumor, stroma, or necrosis), with a tissue mask

    Parameter:
        image_wsi_file : str
           The image file to read. File should be compatiable with openslide.

        tissue_mask_file : str, optional
           The image mask file to read.

        anno_polygons: dict (str: shapely.geometry.(Multi)Polygon) : 
           A dictionary from annotation class name to shapely (Multi)Polygon representing that annotation class.

    Returns:
        dict (str: shapely.geometry.(Multi)Polygon) : a dictionary from annotation class name to shapely (Multi)Polygon representing that annotation class _intersected_ with the mask
    """
    # retain only those annotations within the tissue mask 
    # by intersecting the annotation Multipolygons with the tissue mask
    tissue_polygon = upscale_mask_to_polygon(image_wsi_file, tissue_mask_file)
    
    for k in anno_polygons.keys():
        class_key_name = k
        anno_polygons[class_key_name] = anno_polygons[k].intersection(tissue_polygon)
    
    return anno_polygons   


def create_tissue_resident_annotation_polygons(image_wsi_file, annotation_geojson_file, tissue_mask_file = None, annotation_class_prefix = None):
    """Create a shapely (Multi)Polygon for each annotation class (e.g., tumor, stroma, or necrosis) in QuPath-derived GeoJSON file, optionally intersected with a tissue mask.

    Parameter:
        image_wsi_file : str
           The image file to read. File should be compatiable with openslide.

        annotation_geojson_file : str
           A GeoJSON file of annotations (e.g., as output by QuPath). 
           The annotations are assumed to have named classifications (e.g., tumor, stroma, necrosis).

        tissue_mask_file : str, optional
           The image mask file to read.

        annotation_class_prefix : str, optional
           A string to prepend to the annotation class name in the returned dictionary.

    Returns:
        dict (str: shapely.geometry.(Multi)Polygon) : a dictionary from annotation class name to shapely (Multi)Polygon representing that annotation class.
    """
    
    # read in the annotations
    allobjects = geojson.load(open(annotation_geojson_file))
    feats = [a for a in allobjects["features"] if 'classification' in a['properties'].keys() ]
    
    # separate the annotations by class / name (e.g., tumor, stroma, etc.)
    feature_classes = [feat['properties']['classification']['name'] for feat in feats]
    feat_dict = {}
    for feat in feature_classes:
        feat_dict[feat] = []
    for feat in feats:
        feat_dict[feat['properties']['classification']['name']].append(feat)
    
    # translate the annotations to shapely Polygons, separated by annotation class
    shape_dict = {}
    for k in feat_dict.keys():
        shape_dict[k] = []
        for feat in feat_dict[k]:
            shape_dict[k].append(shape(feat["geometry"]))
    
    # collapse all Polygons of a given class into one MultiPolygon
    anno_polygons = {}
    for k in shape_dict.keys():
        anno_polygons[k] = unary_union(shape_dict[k])
    
    # retain only those annotations within the tissue mask (if specified),
    # by intersecting the annotation Multipolygons with the tissue mask
    if tissue_mask_file is not None:
        tissue_polygon = upscale_mask_to_polygon(image_wsi_file, tissue_mask_file)
    
        for k in anno_polygons.keys():
            class_key_name = k
            if annotation_class_prefix is not None:
                class_key_name = annotation_class_prefix + class_key_name
            anno_polygons[class_key_name] = anno_polygons[k].intersection(tissue_polygon)
    
    return anno_polygons   


def calculate_tile_annotation_properties(tile, annotation_polygon):
    """Return tile coordinates and area of intersection with annotation polygon.

    Parameter:
        tile : shapely.geometry 
           Object representing an image tile.

        annotation_polygon : shapely.geometry.(Multi)Polygon  
           a shapely (Multi)Polygon representing an annotation class.
           
    Returns:
        list(minx, maxx, miny, maxy, area), where
           minx : int
              The minimum x coordinate of the tile; the tile coordinates are [minx, maxx).
           maxx : int
              The maximum x coordinate of the tile; the tile coordinates are [minx, maxx).
           miny : int
              The minimum y coordinate of the tile; the tile coordinates are [miny, maxy).
           maxy : int
              The maximum y coordinate of the tile; the tile coordinates are [miny, maxy).
           area : float
              The fractional area of the tile that intersects the annotation polygon.

    """
    x, y = tile.exterior.coords.xy
    minx = int(min(x))
    maxx = int(max(x))
    miny = int(min(y))
    maxy = int(max(y))
    area = tile.intersection(annotation_polygon).area / tile.area
    return [minx, maxx, miny, maxy, area]


def label_image_tiles(image_wsi_file, annotation_polygons, tile_size = 512, overlap = 0, column_prefix = ''):
    """Label tiles within the image according to their overlap with named annotations.

    Parameter:
        image_wsi_file : str
           The image file to read. File should be compatiable with openslide.

        annotation_polygons: dict (str: shapely.geometry.(Multi)Polygon) : 
           A dictionary from annotation class name to shapely (Multi)Polygon representing that annotation class.

        tile_size : int
           The desired size of the tiles in pixels (in both the x and y dimensions)

        overlap : int
           The number of pixels overlapping contiguous tiles (in both the x and y dimensions)
    
        column_prefix : str
           A string to prepend to the annotation columns returned. For example for annotation 'anno' and label 'lbl', there would be column ***column_prefix***anno_lbl
       
    Returns:
        pandas.DataFrame, where each row corresponds to a tile, and with the following columns:
        - minx : int
             The minimum x coordinate of the tile; the tile coordinates are [minx, maxx).
        - maxx : float
             The maximum x coordinate of the tile; the tile coordinates are [minx, maxx).
        - miny : float
             The minimum y coordinate of the tile; the tile coordinates are [miny, maxy).
        - maxy : float
             The maximum y coordinate of the tile; the tile coordinates are [miny, maxy).
        - ***column_prefix**<anno>_area : float
             For each annotated polygon 'anno', the fractional area of the tile that intersects anno.
        - ***column_prefix**max_area : float
             The maximum over annotations anno of ***column_prefix**<anno>_area
        - ***column_prefix**sum_area : float
             The sum over annotations anno of ***column_prefix**<anno>_area

    """

    # based loosely on https://github.com/choosehappy/QuPathGeoJSONImportExport/blob/master/classify_geojson_objects_in_wsi_centeroid_based_withlevel_mask.py
    # tile the image (in image_wsi_file) with square tiles of size 'tile_size' x 'tile_size' and with 'overlap' overlapping pixels.
    # and annotate its overlap with the annotation polygon

    # open the image and get its dimensions
    slide = openslide.open_slide(image_wsi_file)
    # print('magnification: ' + str(slide.properties['openslide.objective-power']))
    xsz, ysz = slide.dimensions

    # create tiles and put them all in a search tree
    # dz = DeepZoomGenerator(slide, tile_size, overlap, limit_bounds=True)
    tile_polys = []
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    for x in range(0, xsz - tile_size, tile_size - overlap):
        for y in range(0, ysz - tile_size, tile_size - overlap):
            coords = ( (x, y), (x + tile_size, y), (x + tile_size, y + tile_size), (x, y + tile_size) )
            tilepoly = Polygon(coords)
            tile_polys.append(tilepoly)
            xmins.append(x)
            xmaxs.append(x + tile_size)
            ymins.append(y)
            ymaxs.append(y + tile_size)
            
    
    # return tile_polys
    # put tiles in a searchtree, so we can efficiently look for intersections
    # between them and the annotations.
    searchtree = STRtree(tile_polys)

    # for each annotation class, create a DataFrame with all of the tile coordinates.
    # if we don't do this, but only create a DataFrame from the query hits, we won't have
    # entries for all tiles. for consistency, let's have entries for empty ties as well.
    dfs = {}
    for cls in annotation_polygons.keys():
        dfs[cls] = pd.DataFrame({'minx': xmins, 'maxx': xmaxs, 'miny': ymins, 'maxy': ymaxs})
    
    # iterate over each of the annotation classes (e.g., stroma, tumor, necrotic)
    # look for intersections between each class and the tiles
    
    for cls in annotation_polygons.keys():
        hits = searchtree.query(annotation_polygons[cls])
        # hits = [hit for hit in hits if hit.intersection(annotation_polygons[cls]).area / hit.area > positive_annotation_cutoff]
        rows = [calculate_tile_annotation_properties(hit, annotation_polygons[cls]) for hit in hits]
        # dfs[cls] = pd.DataFrame(rows, columns = ['minx', 'maxx', 'miny', 'maxy', column_prefix + cls + '_area'])
        # these next lines replace the former and ensure we have entries for empty tiles as well.
        tmp = pd.DataFrame(rows, columns = ['minx', 'maxx', 'miny', 'maxy', column_prefix + cls + '_area'])
        dfs[cls] = dfs[cls].merge(tmp, how='outer', on = ['minx', 'maxx', 'miny', 'maxy']).fillna(0.0)
    
    for i,j in dfs.items():
        print(str(i) + " has " + str(len(j)) + " tiles")
    dfs = {i : j.set_index(['minx','maxx','miny', 'maxy']) for i,j in dfs.items()}
    tbl = pd.concat(dfs.values(),axis=1)
    tbl = tbl.reset_index(drop = False)
    tbl = tbl.fillna(0)
    
    tbl['tile'] = tbl[['minx', 'miny']].apply(lambda x: str(int(x[0] / tile_size)) + "_" + str(int(x[1] / tile_size)), axis = 1)
    
    classes = [cls for cls in annotation_polygons.keys()]
    class_areas = [column_prefix + cls + '_area' for cls in annotation_polygons.keys()]
    if len(annotation_polygons) > 1:
        tbl[column_prefix + 'max_area'] = tbl[class_areas].apply(lambda x: max(x), axis=1)
        tbl[column_prefix + 'sum_area'] = tbl[class_areas].apply(lambda x: sum(x), axis=1)
        # tbl[column_prefix + 'classification'] = tbl[class_areas].apply(lambda x: classes[np.argmax(x)], axis=1)

    return tbl


def label_image_tile_region_and_tissue(image_wsi_file, annotation_geojson_file, tissue_mask_file = None, tile_size = 512, overlap = 0):
    """Label tiles within the image according to their overlap with: (1) named annotations (e.g., tumor, stroma, necrosis), 
 named annotations.

    Parameter:
        image_wsi_file : str
           The image file to read. File should be compatiable with openslide.

        annotation_geojson_file : str
           A GeoJSON file of annotations (e.g., as output by QuPath). 
           The annotations are assumed to have named classifications (e.g., tumor, stroma, necrosis).

        tissue_mask_file : str, optional
           The image mask file to read.

        tile_size : int
           The desired size of the tiles in pixels (in both the x and y dimensions)

        overlap : int
           The number of pixels overlapping contiguous tiles (in both the x and y dimensions)
    
    Returns:
        pandas.DataFrame, where each row corresponds to a tile, and with the following columns:
        - minx : int
             The minimum x coordinate of the tile; the tile coordinates are [minx, maxx).
        - maxx : float
             The maximum x coordinate of the tile; the tile coordinates are [minx, maxx).
        - miny : float
             The minimum y coordinate of the tile; the tile coordinates are [miny, maxy).
        - maxy : float
             The maximum y coordinate of the tile; the tile coordinates are [miny, maxy).
        - region_<anno>_area : float
             For each annotated region 'anno', the fractional area of the tile that intersects anno.
        - region_area : float
             The maximum over annotations anno of region_<anno>_area
        - region_sum_area : float
             The sum over annotations anno of region_<anno>_area
        - tissue_area : float
             The fractional area of the tile that intersects the tissue mask if tissue_mask_file is not None.
        - tissue_region_<anno>_area : float
             For each annotated region 'anno', the fractional area of the tile that intersects anno _and_ the tissue mask if tissue_mask_file is not None.
        - tissue_region_area : float
             The maximum over annotations anno of tissue_region_<anno>_area, if tissue_mask_file is not None.
        - tissue_region_sum_area : float
             The sum over annotations anno of tissue_region_<anno>_area, if tissue_mask_file is not None.
    """

    # Define the annotation polygons from the GeoJSON file
    annotation_polygons = create_annotation_polygons(annotation_geojson_file)

    # Label each image tile with its fractional overlap with each annotation polygon
    tbl = label_image_tiles(image_wsi_file, annotation_polygons, tile_size = tile_size, overlap = overlap, column_prefix = 'region_')

    if tissue_mask_file is not None:

        # Define the tissue polygon from the input tissue mask
        tissue_polygons = upscale_mask_to_polygon(image_wsi_file, tissue_mask_file)

        # Label each image tile with its fractional overlap with the tissue mask/polygon
        tbl2 = label_image_tiles(image_wsi_file, {'tissue': tissue_polygons}, tile_size = tile_size, overlap = overlap, column_prefix = '')
        tbl = tbl.merge(tbl2, how='outer', on = ['minx', 'maxx', 'miny', 'maxy', 'tile']).fillna(0.0)

        # Intersect the annotation polygons with the tissue mask
        tissue_resident_annotation_polygons = intersect_annotation_polygons_with_mask(image_wsi_file, tissue_mask_file, annotation_polygons)

        # Label each image tile with its fractional overlap with _both_ the tissue mask and the annotation polygons
        # e.g., for expediency a "cancer" region annotation might spill outside the tissue. This would ensure that
        # regions only annotate the tissue.
        tbl2 = label_image_tiles(image_wsi_file, tissue_resident_annotation_polygons, tile_size = tile_size, overlap = overlap, column_prefix = 'tissue_region_')        
        tbl = tbl.merge(tbl2, how='outer', on = ['minx', 'maxx', 'miny', 'maxy', 'tile']).fillna(0.0)
        
    return tbl


