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

def filter_hovernet_nuclei_by_tile_bounds(bbox_list, centroid_list, contour_list, type_list, xmin, ymin, width, height, scale_factor = 1):
    """Filter the nuclei bounded by a tile and remap the coordinates of the nuclei relative to the top left of the tile

    Parameter:
        bbox_list (list): bounding boxes for each nucleus
        centroid_list (list): centroids for each nucleus
        contour_list (list): contours for each nucleus
        type_list (list): types associated with each nucleus
        xmin, ymin (int): coordinates of top left corner of tile
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
            bbox_tmp = np.array(bbox_list[idx])
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
          - cell_id (uint32): the id of the cell
          - centroid_x, centroid_y (floats): the x and y coordinates of the cell's centroid
          - cell_type (uint8): cell type identifier to be cross referenced with HoverNet type_info.json file 
          - cell_type_area (float): area of cell
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
          - cell_id (uint32): the id of the cell
          - centroid_x, centroid_y (floats): the x and y coordinates of the cell's centroid
          - cell_type (uint8): cell type identifier to be cross referenced with HoverNet type_info.json file 
          - cell_type_area (float): area of cell
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

def compute_tabular_hovernet_results(hovernet_base_dir):
    """Extract cell info (e.g., centroid location and cell type) from all Hover-Net output files in hovernet_base_dir

    Parameter:
        hovernet_base_dir (str): directory containing Hover-Net results for image <x> in sub-directory <hovernet_base_dir>/<x>_wsi/json/<x>.json

    Returns:
        a DataFrame with columns:
          - cell_id (uint32): the id of the cell
          - centroid_x, centroid_y (floats): the x and y coordinates of the cell's centroid
          - cell_type (uint8): cell type identifier to be cross referenced with HoverNet type_info.json file 
          - cell_type_area (float): area of cell
          - image_id (string): the name of the image
    """
    hovernet_image_ids = [x for x in listdir(hovernet_base_dir) if x.endswith('_wsi')]
    hovernet_image_ids = [x.replace('_wsi', '') for x in hovernet_image_ids]
    print('Extracting cell info from hovernet')
    dfs = [extract_cell_info_from_hovernet_output(hovernet_base_dir + x + '_wsi/' + 'json/' + x + '.json')[0] for x in hovernet_image_ids]
    print('Creating hovernet DataFrame')
    hov_df = pd.concat(dfs, keys = hovernet_image_ids).reset_index().drop("level_1", axis=1).rename(columns = {"level_0": "image_id", "ids": "cell_id"})
    return hov_df

def output_tabular_hovernet_results(hovernet_base_dir, output_file_prefix):
    """Extract cell info (e.g., centroid location and cell type) from all Hover-Net output files in hovernet_base_dir and store all results in a single file <output_file_prefix>-hovernet-centroids.csv

    Parameter:
        hovernet_base_dir (str): directory containing Hover-Net results for image <x> in sub-directory <hovernet_base_dir>/<x>_wsi/json/<x>.json
        output_file_prefix (str): prefix of file in which to store results (<output_file_prefix>-hovernet-centroids.csv)

    Returns:
        Nothing.
        Writes a csv file <output_file_prefix>-hovernet-centroids.csv with the following columns:
          - cell_id (uint32): the id of the cell
          - centroid_x, centroid_y (floats): the x and y coordinates of the cell's centroid
          - cell_type (uint8): cell type identifier to be cross referenced with HoverNet type_info.json file 
          - cell_type_area (float): area of cell
          - image_id (string): the name of the image
    """
    hov_df = compute_tabular_hovernet_results(hovernet_base_dir)
    print('Outputing hovernet DataFrame')
    hov_df.to_csv(output_file_prefix + "-hovernet-centroids.csv", index=False)

def compute_tabular_tissue_mask_results(histoqc_base_dir, image_postfix = 'svs'):
    """Compute number of non-zero pixels in all tissue masks within subdirectories of <histoqc_base_dir>, i.e., those files for image <x> in <histoqc_base_dir>/<x><image_postfix>/<x><image_postix>_mask_use.png

    Parameter:
        histoqc_base_dir (str): directory containing histoqc tissue masks results for image <x> in sub-directory <histoqc_base_dir>/<x><image_postfix>/<x><image_postix>_mask_use.png
        image_postfix (str): the postfix used to find the images

    Returns:
        a DataFrame with columns:
          - image (string): the name of the image
          - mask_nz_pizels (int): the number of non-zero pixels in the tissue mask (at the resolution of the mask, which is 1.25X for histoqc, and not the resolution of the original image)
    """
    wsi_files = [x for x in listdir(histoqc_base_dir) if x.endswith(image_postfix)]
    print('Counting non zeroes')
    cnts = [count_non_zero_pixels(histoqc_base_dir + '/' + wsi_file + '/' + wsi_file + '_mask_use.png') for wsi_file in wsi_files]
    print('Creating non-zero pixel DataFrame')
    df = pd.DataFrame({'image': wsi_files, 'mask_nz_pixels': cnts})
    return df 

def output_tabular_results(histoqc_base_dir, hovernet_base_dir, output_file_prefix, image_postfix = 'svs'):
    """Extract cell info (e.g., centroid location and cell type) from all Hover-Net output files in hovernet_base_dir and store all results in a single file <output_file_prefix>-hovernet-centroids.csv. Also store the number of non-zero pixels in the tissue masks for each of those images in a single file <output_file_prefix>-mask-area.csv

    Parameter:
        hovernet_base_dir (str): directory containing Hover-Net results for image <x> in sub-directory <hovernet_base_dir>/<x>_wsi/json/<x>.json
        output_file_prefix (str): prefix of file in which to store results (<output_file_prefix>-hovernet-centroids.csv)
        histoqc_base_dir (str): directory containing histoqc tissue masks results for image <x> in sub-directory <histoqc_base_dir>/<x><image_postfix>/<x><image_postix>_mask_use.png
        image_postfix (str): the postfix used to find the images

    Returns:
        Nothing.

        Writes a csv file <output_file_prefix>-hovernet-centroids.csv with the following columns:
          - cell_id (uint32): the id of the cell
          - centroid_x, centroid_y (floats): the x and y coordinates of the cell's centroid
          - cell_type (uint8): cell type identifier to be cross referenced with HoverNet type_info.json file 
          - cell_type_area (float): area of cell
          - image_id (string): the name of the image

        Writes a csv file <output_file_prefix>-mask-area.csv with the following columns:
          - image (string): the name of the image
          - mask_nz_pizels (int): the number of non-zero pixels in the tissue mask (at the resolution of the mask, which is 1.25X for histoqc, and not the resolution of the original image)
    """
    wsi_files = [x for x in listdir(histoqc_base_dir) if x.endswith(image_postfix)]
    print('Counting non zeroes')
    print(len(wsi_files))
    cnts = [count_non_zero_pixels(histoqc_base_dir + '/' + wsi_file + '/' + wsi_file + '_mask_use.png') for wsi_file in wsi_files]
    print('Creating non-zero pixel DataFrame')
    df = pd.DataFrame({'image': wsi_files, 'mask_nz_pixels': cnts})
    print('Outputing mask area DataFrame')
    df.to_csv(output_file_prefix + "-mask-area.csv", index=False)
    hovernet_image_ids = [x for x in listdir(hovernet_base_dir) if x.endswith('_wsi')]
    hovernet_image_ids = [x.replace('_wsi', '') for x in hovernet_image_ids]
    print(len(hovernet_image_ids))
    print('Extracting cell info from hovernet')
    dfs = [extract_cell_info_from_hovernet_output(hovernet_base_dir + x + '_wsi/' + 'json/' + x + '.json')[0] for x in hovernet_image_ids]
    print('Creating hovernet DataFrame')
    hov_df = pd.concat(dfs, keys = hovernet_image_ids).reset_index().drop("level_1", axis=1).rename(columns = {"level_0": "image_id", "ids": "cell_id"})
    print('Outputing hovernet DataFrame')
    hov_df.to_csv(output_file_prefix + "-hovernet-centroids.csv.gz", index=False)

def output_tabular_results_per_file(histoqc_base_dir, hovernet_base_dir, output_file_prefix, image_postfix = 'svs'):
    """Extract cell info (e.g., centroid location and cell type) from all Hover-Net results corresponding to image <x> in <hovernet_base_dir> and store results for each image in file <output_file_prefix>-<x>-hovernet-centroids.csv.gz. Also store the number of non-zero pixels in the tissue masks for each of those images in a single file <output_file_prefix>-mask-area.csv

    Parameter:
        histoqc_base_dir (str): directory containing histoqc tissue masks results for image <x> in sub-directory <histoqc_base_dir>/<x><image_postfix>/<x><image_postix>_mask_use.png
        hovernet_base_dir (str): directory containing Hover-Net results for image <x> in sub-directory <hovernet_base_dir>/<x>_wsi/json/<x>.json
        output_file_prefix (str): prefix of file in which to store results (<output_file_prefix>-hovernet-centroids.csv)
        image_postfix (str): the postfix used to find the images

    Returns:
        Nothing.

        Writes gzipped csv files <output_file_prefix>-<x>-hovernet-centroids.csv for each image <x> with the following columns:
          - cell_id (uint32): the id of the cell
          - centroid_x, centroid_y (floats): the x and y coordinates of the cell's centroid
          - cell_type (uint8): cell type identifier to be cross referenced with HoverNet type_info.json file 
          - cell_type_area (float): area of cell
          - image_id (string): the name of the image

        Writes a csv file <output_file_prefix>-mask-area.csv with the following columns:
          - image (string): the name of the image
          - mask_nz_pizels (int): the number of non-zero pixels in the tissue mask (at the resolution of the mask, which is 1.25X for histoqc, and not the resolution of the original image)
    """
    wsi_files = [x for x in listdir(histoqc_base_dir) if x.endswith(image_postfix)]
    print('Counting non zeroes')
    print(len(wsi_files))
    cnts = [count_non_zero_pixels(histoqc_base_dir + '/' + wsi_file + '/' + wsi_file + '_mask_use.png') for wsi_file in wsi_files]
    print('Creating non-zero pixel DataFrame')
    df = pd.DataFrame({'image': wsi_files, 'mask_nz_pixels': cnts})
    print('Outputing mask area DataFrame')
    df.to_csv(output_file_prefix + "-mask-area.csv", index=False)
    hovernet_image_ids = [x for x in listdir(hovernet_base_dir) if x.endswith('_wsi')]
    hovernet_image_ids = [x.replace('_wsi', '') for x in hovernet_image_ids]
    print('Extracting cell info from hovernet')
    for x in hovernet_image_ids:
        output_file = output_file_prefix + '-' + x + '-hovernet-centroids.csv.gz'
        if not exists(output_file):
            print(x)
            df = extract_cell_info_from_hovernet_output(hovernet_base_dir + x + '_wsi/' + 'json/' + x + '.json')[0]
            df = df.rename(columns = {"ids": "cell_id"})
            df['image_id'] = x
            df.to_csv(output_file, index=False)
    print('Done extracting cell info from hovernet')

def output_tabular_results_per_file_(hovernet_base_dir, hovernet_image_ids, output_file_prefix, use_ijson = False):
    """Extract cell info (e.g., centroid location and cell type) from all Hover-Net results corresponding to image <x> in <hovernet_base_dir> and store results for each image in file <output_file_prefix>-<x>-hovernet-centroids.csv.gz. 

    Parameter:
        hovernet_base_dir (str): directory containing Hover-Net results for image <x> in sub-directory <hovernet_base_dir>/<x>_wsi/json/<x>.json
        hovernet_image_ids (list of strings): list holding subset of images to process within hovernet_base_dir or None, if all should be processed
        output_file_prefix (str): prefix of file in which to store results (<output_file_prefix>-hovernet-centroids.csv)
        use_ijson (boolean): whether to parse Hover-Net files in a memory-efficient, but slow, manner using ijson.

    Returns:
        Nothing.

        Writes gzipped csv files <output_file_prefix>-<x>-hovernet-centroids.csv for each image <x> with the following columns:
          - cell_id (uint32): the id of the cell
          - centroid_x, centroid_y (floats): the x and y coordinates of the cell's centroid
          - cell_type (uint8): cell type identifier to be cross referenced with HoverNet type_info.json file 
          - cell_type_area (float): area of cell
          - image_id (string): the name of the image
    """
    ids = [x for x in listdir(hovernet_base_dir) if x.endswith('_wsi')]
    ids = [x.replace('_wsi', '') for x in ids]
    if hovernet_image_ids is not None:
        ids = [x for x in ids if x in hovernet_image_ids]
    print('Extracting cell info from hovernet: ' + str(len(ids)) + ' files')
    for x in ids:
        output_file = output_file_prefix + '-' + x + '-hovernet-centroids.csv.gz'
        if not exists(output_file):
            print(x)
            if use_ijson:
                df = extract_cell_info_from_hovernet_output_ijson(hovernet_base_dir + x + '_wsi/' + 'json/' + x + '.json')[0]
            else:
                df = extract_cell_info_from_hovernet_output(hovernet_base_dir + x + '_wsi/' + 'json/' + x + '.json')[0]
            df = df.rename(columns = {"ids": "cell_id"})
            df['image_id'] = x
            df.to_csv(output_file, index=False)
    print('Done extracting cell info from hovernet')
