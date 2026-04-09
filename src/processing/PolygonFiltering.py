
import numpy as np
from math import hypot, degrees, atan2, radians, cos, sin
from shapely.geometry import Polygon
import itertools

def reshape_and_offset_polygon(polygon, ratios, offset, size = None):
    polygon = np.asarray(polygon).reshape(-1,2)
    polygon[:,0] = ((polygon[:,0]-offset)*ratios[0])
    polygon[:,1] = (polygon[:,1]*ratios[1])
    if size is not None:
        polygon[:,0]/=size[0]
        polygon[:,1]/=size[1]
    return polygon

def _calculate_obb_corners(cx, cy, width, height, angle_degrees):
    """Helper function to calculate OBB corners."""
    angle_rad = radians(float(angle_degrees))
    hw = width / 2.0
    hh = height / 2.0
    
    corners_relative_to_origin = [
        (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
    ]
    
    rotated_corners = []
    for x_rel, y_rel in corners_relative_to_origin:
        x_rot = x_rel * cos(angle_rad) - y_rel * sin(angle_rad)
        y_rot = x_rel * sin(angle_rad) + y_rel * cos(angle_rad)
        rotated_corners.append((x_rot + cx, y_rot + cy))
    # rotated_corners = list(itertools.chain(*rotated_corners))
    return [(round(x),round(y)) for x,y in rotated_corners]

def order_rectangle_coords(coords):
    """
    Placeholder pour la fonction order_rectangle_coords de l'utilisateur.
    Doit ordonner les coordonnées d'un rectangle.
    """
    return coords

def polygon_to_limited_bbox(polygon_input, ratios=(1,1), offset=0, max_height=30):
    """
    Convertit un polygone en une représentation de boîte englobante xmin, ymin, xmax, ymax, et angle.
    Limite la hauteur de la boîte englobante si max_height est fourni.

    Args:
        polygon_input (list): Coordonnées du polygone d'entrée.
        ratios (tuple): Ratios pour le remodelage.
        offset (float/int): Décalage pour le remodelage.
        max_height (float, optional): Hauteur maximale de la boîte englobante.

    Returns:
        list: Une liste contenant [xmin, ymin, xmax, ymax, angle] de l'AABB de l'OBB.
              Les coordonnées sont des flottants, l'angle est un entier.
    """
    # Copier explicitement les coordonnées pour éviter la modification de l'original
    polygon_input = np.array(polygon_input, dtype=float).reshape(-1,2).copy()
    polygon_processed = reshape_and_offset_polygon(polygon_input, ratios, offset)
    polygon_shape = Polygon(polygon_processed)
    obb = polygon_shape.minimum_rotated_rectangle
    obb_coords_raw = np.array(obb.exterior.coords)
    obb_coords = obb_coords_raw[:-1]
    ordered_obb_coords = order_rectangle_coords(obb_coords)
    if polygon_input.shape[0] != 4:
        centroid_poly = polygon_shape.centroid
    else:
        centroid_poly = Polygon(ordered_obb_coords).centroid
    cx = centroid_poly.x
    cy = centroid_poly.y

    edge1_len = hypot(ordered_obb_coords[0][0] - ordered_obb_coords[1][0], 
                      ordered_obb_coords[0][1] - ordered_obb_coords[1][1])
    edge2_len = hypot(ordered_obb_coords[1][0] - ordered_obb_coords[2][0], 
                      ordered_obb_coords[1][1] - ordered_obb_coords[2][1])

    dx_edge1 = ordered_obb_coords[1][0] - ordered_obb_coords[0][0]
    dy_edge1 = ordered_obb_coords[1][1] - ordered_obb_coords[0][1]
    angle_rad_edge1 = atan2(dy_edge1, dx_edge1)
    angle_deg_edge1 = degrees(angle_rad_edge1)

    # `width_for_angle` est la dimension dont l'orientation est `final_angle`
    # Initialement, `width_for_angle` est le côté le plus long de l'OBB.
    if edge1_len >= edge2_len:
        width_for_angle = edge1_len
        height_perp_to_angle = edge2_len
        final_angle_deg = angle_deg_edge1
    else:
        width_for_angle = edge2_len
        height_perp_to_angle = edge1_len
        final_angle_deg = angle_deg_edge1 + 90
    
    if final_angle_deg > 90:
        final_angle_deg -= 180
    elif final_angle_deg < -90:
        final_angle_deg += 180
    
    final_angle_int = round(final_angle_deg)

    # Appliquer max_height à la dimension 'hauteur' (height_perp_to_angle)
    actual_height = height_perp_to_angle
    if max_height is not None and height_perp_to_angle > max_height:
        actual_height = float(max_height)
        print(f"polygon_to_limited_bbox: Hauteur limitée à {actual_height:.2f} (originale: {height_perp_to_angle:.2f})")

    # `width_for_angle` et `actual_height` sont les dimensions finales de l'OBB.
    # `final_angle_int` est l'orientation de `width_for_angle`.
    print(f"polygon_to_limited_bbox: OBB final - largeur={width_for_angle:.2f}, hauteur={actual_height:.2f}, angle={final_angle_int}")
    
    # Calculer les coins de cet OBB final
    obb_final_corners = _calculate_obb_corners(cx, cy, width_for_angle, actual_height, final_angle_int)
    
    # Calculer l'AABB de ces coins
    corners_np = np.array(obb_final_corners)
    xmin = np.min(corners_np[:, 0])
    xmax = np.max(corners_np[:, 0])
    ymin = np.min(corners_np[:, 1])
    ymax = np.max(corners_np[:, 1])
    
    # Retourner [xmin, ymin, xmax, ymax, angle]
    bbox = [round(xmin), round(ymin), round(xmax), round(ymax), final_angle_int]
    return bbox

def polygon_to_centroid_bbox(polygon, ratios, offset):
    polygon = np.array(polygon).reshape(-1,2)
    polygon[:,0] = ((polygon[:,0]-offset)*ratios[0])#/size[0]
    polygon[:,1] = (polygon[:,1]*ratios[1])#/size[1]
    polygon = Polygon(polygon)
    x_min, y_min, x_max, y_max = polygon.bounds
    centroid = polygon.centroid

    bbox = [x_min, max(y_min,centroid.y-20), x_max, min(y_max,centroid.y+30)]
    bbox = [round(bbox[0]), round(bbox[1]), round(bbox[2]), round(bbox[3])]
    return bbox

def sort_polygons(polygons):
    """Trie les polygones de gauche à droite et de haut en bas."""
    def mean_height(polygon):
        return np.mean([point[1] for point in np.array(polygon["points"]).reshape(-1,2)])
    sorted_polygons = sorted(polygons, key=mean_height)
    return sorted_polygons