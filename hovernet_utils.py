import json

def summarize_hovernet_cells(json_path):
    """Output summary statistics of cells in Hover-Net JSON file.

    Parameter:
        json_path (str): path to json file output by Hover-Net

    Returns:
        a tuple containing

        - df (DataFrame): 
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
