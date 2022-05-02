import json
# conda -c conda-forge ijson
import ijson
import numpy as np
import pandas as pd

def polygon_area(x,y):
    """Calculate the area of polygon with contours specified as ***x*** and ***y*** coordinates.

    Parameter:
        x (np.array(dtype=float, ndim=1)): x coordinates of polygon contour
        y (np.array(dtype=float, ndim=1)): y coordinates of polygon contour

    Returns:
        float: the area of the polygon with the given contour
    """
    # An alternative implementation:
    # from shapely.geometry import Polygon
    # pgon = Polygon(zip(x, y))
    # return pgon.area
    
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)

def summarize_hovernet_cells(json_path):
    """Output summary statistics of cells in Hover-Net JSON file.

    Parameter:
        json_path (str): path to json file output by Hover-Net

    Returns:
        a tuple containing

        - df (DataFrame): a DataFrame with columns cell_type (number to
          be cross referenced with HoverNet type_info.json file), cell_type_area
          (the total area of all cells of cell_type), and cell_type_counts
          (the total number of cells of cell_type).
        - mag (float): the magnification of the image Hover-Net was run on
    """
    count_dict = {}
    area_dict = {}

    with open(json_path) as json_file:
        data = json.load(json_file)
        mag_info = data['mag']
        nuc_info = data['nuc']
        for inst in nuc_info:
            inst_info = nuc_info[inst]
            # inst_centroid = inst_info['centroid']
            # centroid_list.append(inst_centroid)
            inst_contour = inst_info['contour']
            # contour_list.append(inst_contour)
            # inst_bbox = inst_info['bbox']
            # bbox_list.append(inst_bbox)
            inst_type = inst_info['type']
            if not inst_type in count_dict:
                count_dict[inst_type] = 0
                area_dict[inst_type] = 0
            count_dict[inst_type] = count_dict[inst_type] + 1
            x=np.asarray(inst_contour)[:,0]
            y=np.asarray(inst_contour)[:,1]
            area = polygon_area(x, y)
            area_dict[inst_type] = area_dict[inst_type] + area
            # type_list.append(inst_type)
        
    keys = list(area_dict.keys())
    areas = [area_dict[k] for k in keys]
    counts = [count_dict[k] for k in keys]
    df = pd.DataFrame({"cell_type": keys, "cell_type_area": areas, "cell_type_counts": counts})
    return df, mag_info

def extract_cell_info_lists_from_hovernet_output(json_path):
    """Extract cell info (e.g., centroid location and cell type) from Hover-Net JSON output.

    Parameter:
        json_path (str): path to json file output by Hover-Net

    Returns:
        - mag (float): the magnification of the image Hover-Net was run on
        - bbox_list (list): bounding boxes for each nucleus
        - centroid_list (list): centroids for each nucleus
        - contour_list (list): contours for each nucleus
        - type_list (list): types associated with each nucleus
    """
    json_file = open(json_path)
    data = json.load(json_file)

    mag_info = data['mag']
    nuc_info = data['nuc']
    n_cells = len(nuc_info)

    print('n_cells: ' + str(n_cells))

    bbox_list = []
    centroid_list = []
    contour_list = [] 
    type_list = []

    for inst in nuc_info:
        inst_info = nuc_info[inst]
        inst_centroid = inst_info['centroid']
        centroid_list.append(inst_centroid)
        inst_contour = inst_info['contour']
        contour_list.append(inst_contour)
        inst_bbox = inst_info['bbox']
        bbox_list.append(inst_bbox)
        inst_type = inst_info['type']
        type_list.append(inst_type)

    return mag_info, bbox_list, centroid_list, contour_list, type_list

def filter_hovernet_nuclei_by_tile_bounds(bbox_list, centroid_list, contour_list, type_list, xmin, xmax, width, height, scale_factor = 1):
    """Filter the nuclei bounded by a tile and remap the coordinates of the nuclei relative to the top left of the tile

    Parameter:
        bbox_list (list): bounding boxes for each nucleus
        centroid_list (list): centroids for each nucleus
        contour_list (list): contours for each nucleus
        type_list (list): types associated with each nucleus
        xmin, xmax (int): coordinates of top left corner of tile
        width, height (int): width and height of tile
        scale_factor (float): ratio by which to _down_sample coordinates (i.e., scale of coordinates of nuclei relative to original image)

    Returns:
        - bbox_list (list): bounding boxes for each nucleus within tile, scaled and relative to tile top left corner
        - centroid_list (list): centroids for each nucleus within tile, scaled and relative to tile top left corner
        - contour_list (list): contour_list for each nucleus within tile, scaled and relative to tile top left corner
        - type_list (list): type for each nucleus within tile

    """

    filtered_bbox_list = []
    filtered_centroid_list = []
    filtered_contour_list = [] 
    filtered_type_list = []

    coords_xmin = xmin
    coords_xmax = xmin + width
    coords_ymin = ymin
    coords_ymax = ymin + height
    
    for idx, cnt in enumerate(contour_list):
        cnt_tmp = np.array(cnt)
        cnt_tmp = cnt_tmp[(cnt_tmp[:,0] >= coords_xmin) & (cnt_tmp[:,0] <= coords_xmax) & (cnt_tmp[:,1] >= coords_ymin) & (cnt_tmp[:,1] <= coords_ymax)] 
        if cnt_tmp.shape[0] > 0:
            label = str(type_list[idx])
            cnt_adj = np.round((cnt_tmp - np.array([xmin, ymin])) / scale_factor).astype('int')
            centroid = centroid_list[idx]
            centroid_adj = ( centroid - np.array([xmin, ymin]) ) / scale_factor
            bbox_tmp = np.array(bbox_list[indx])
            bbox_adj = np.round((bbox_tmp - np.array([xmin, ymin])) / scale_factor).astype('int')
            filtered_contour_list.append(cnt_adj)
            filtered_centroid_list.append(centroid_adj)
            filtered_type_list.append(label)
    
    return filtered_bbox_list, filtered_centroid_list, filtered_contour_list, filtered_type_list

def extract_cell_info_from_hovernet_output(json_path):
    """Extract cell info (e.g., centroid location and cell type) from Hover-Net JSON output.

    Parameter:
        json_path (str): path to json file output by Hover-Net

    Returns:
        a tuple containing

        - df (DataFrame): a DataFrame with columns cell_type (number to
          be cross referenced with HoverNet type_info.json file), cell_type_area
          (the total area of all cells of cell_type), and cell_type_counts
          (the total number of cells of cell_type).
        - mag (float): the magnification of the image Hover-Net was run on
    """
    json_file = open(json_path)
    data = json.load(json_file)

    mag_info = data['mag']
    nuc_info = data['nuc']
    n_cells = len(nuc_info)

    # lists to store centroid x and y coordinates, cell type, cell area, and cell id
    print('n_cells: ' + str(n_cells))
    centroid_xs = np.empty((n_cells,), dtype=float)
    centroid_ys = np.empty((n_cells,), dtype=float)
    cell_types = np.empty((n_cells,), dtype=np.uint8)
    cell_areas = np.empty((n_cells,), dtype=float)
    ids = np.empty((n_cells,), dtype=np.uint32)
    
    keys = list(nuc_info.keys())
    for i in range(0, n_cells):
        inst_info = nuc_info[keys[i]]
        inst_centroid = inst_info['centroid']
        inst_contour = inst_info['contour']
        inst_type = inst_info['type']
        x=np.asarray(inst_contour)[:,0]
        y=np.asarray(inst_contour)[:,1]
        area = polygon_area(x, y)
        centroid_xs[i] = inst_centroid[0]
        centroid_ys[i] = inst_centroid[1]
        cell_types[i] = inst_type
        cell_areas[i] = area
        # NB: keys[i] is a str (representing an integer)
        ids[i] = np.uint32(keys[i])

    df = pd.DataFrame({"ids": ids, "centroid_x": centroid_xs, "centroid_y": centroid_ys, "cell_type": cell_types, "cell_type_area": cell_areas})
    return df, mag_info

def extract_cell_info_from_hovernet_output_ijson(json_path):
    """Extract cell info (e.g., centroid location and cell type) from Hover-Net JSON output in a memory-efficient manner.

    Parameter:
        json_path (str): path to json file output by Hover-Net

    Returns:
        a tuple containing

        - df (DataFrame): a DataFrame with columns cell_type (number to
          be cross referenced with HoverNet type_info.json file), cell_type_area
          (the total area of all cells of cell_type), and cell_type_counts
          (the total number of cells of cell_type).
        - mag (float): the magnification of the image Hover-Net was run on
    """
    # json_file = open(json_path)
    # data = ijson.parse(json_file)
    data = ijson.parse(open(json_path))
    n_cells = sum(1 for x in ijson.kvitems(data, 'nuc'))

    data = ijson.parse(open(json_path))
    mag = ijson.items(data, 'mag')
    mag_info = next(mag)
    
    # lists to store centroid x and y coordinates, cell type, cell area, and cell id
    print('n_cells: ' + str(n_cells))
    centroid_xs = np.empty((n_cells,), dtype=float)
    centroid_ys = np.empty((n_cells,), dtype=float)
    cell_types = np.empty((n_cells,), dtype=np.uint8)
    cell_areas = np.empty((n_cells,), dtype=float)
    ids = np.empty((n_cells,), dtype=np.uint32)

    # data = ijson.parse(json_file)
    data = ijson.parse(open(json_path))
    nuc = ijson.kvitems(data, 'nuc')
    i = 0
    for k, inst_info in nuc:
        inst_centroid = inst_info['centroid']
        inst_contour = inst_info['contour']
        inst_type = inst_info['type']
        x=np.asarray(inst_contour)[:,0]
        y=np.asarray(inst_contour)[:,1]
        area = polygon_area(x, y)
        centroid_xs[i] = inst_centroid[0]
        centroid_ys[i] = inst_centroid[1]
        cell_types[i] = inst_type
        cell_areas[i] = area
        # NB: keys[i] is a str (representing an integer)
        ids[i] = np.uint32(k)
        i = i + 1

    df = pd.DataFrame({"ids": ids, "centroid_x": centroid_xs, "centroid_y": centroid_ys, "cell_type": cell_types, "cell_type_area": cell_areas})
    return df, mag_info

def extract_hovernet_cell_info(json_path):
    """Extract cell type info from Hover-Net type_info.json file

    Parameter:
        json_path (str): path to type_info.json file input to Hover-Net

    Returns:
        DataFrame with columns:
        - id (object): the cell type id (e.g., as included in the Hover-Net JSON output)
        - label (object): the label of the cell type (e.g., neopla, etc)
        - r_color: the red channel of the RGB color specification used by Hover-Net for this cell type
        - g_color: the green channel of the RGB color specification used by Hover-Net for this cell type
        - b_color: the blue channel of the RGB color specification used by Hover-Net for this cell type
    """

    json_file = open(json_path)
    cell_dict = json.load(json_file)
    
    cell_tbl = pd.DataFrame.from_dict(cell_dict, orient="index", columns = ['label', 'rgb'])
    cell_tbl['r_color'] = cell_tbl.rgb.apply(lambda x: x[0])
    cell_tbl['g_color'] = cell_tbl.rgb.apply(lambda x: x[1])
    cell_tbl['b_color'] = cell_tbl.rgb.apply(lambda x: x[1])
    cell_tbl = cell_tbl.drop(labels=['rgb'], axis=1)
    cell_tbl = cell_tbl.reset_index().rename(columns={'index':'id'})

    return cell_tbl
    
