from .PolygonFiltering import _calculate_obb_corners
from math import radians, cos, sin
import re
import numpy as np
from typing import List, Dict


def bounding_box_to_polygon(xmin, ymin, xmax, ymax, angle_degrees):
    """
    Calcule les quatre coordonnées d'un polygone (rectangle orienté) à partir de
    xmin, ymin, xmax, ymax de son AABB et de son angle de rotation.

    Args:
        xmin (float): Coordonnée x minimale de l'AABB.
        ymin (float): Coordonnée y minimale de l'AABB.
        xmax (float): Coordonnée x maximale de l'AABB.
        ymax (float): Coordonnée y maximale de l'AABB.
        angle_degrees (int): Angle de rotation de l'OBB en degrés.

    Returns:
        list: Liste de tuples représentant les quatre coins du rectangle orienté.
    """
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    W_aabb = float(xmax - xmin)
    H_aabb = float(ymax - ymin)
    
    angle_rad = radians(float(angle_degrees))
    angle_rad = radians(float(angle_degrees))
    cos_a = cos(angle_rad)
    sin_a = sin(angle_rad)
    
    # Utiliser les valeurs absolues pour c et s dans les formules de reconstruction
    abs_c = abs(cos_a)
    abs_s = abs(sin_a)
    
    # Dénominateur pour la résolution du système d'équations
    # det = cos²(angle_rad) - sin²(angle_rad) mais en utilisant abs_c et abs_s pour la formule W = wc+hs, H = ws+hc
    # det = abs_c*abs_c - abs_s*abs_s
    # Les formules sont w_obj = (abs_c*W_aabb - abs_s*H_aabb) / det et h_obj = (abs_c*H_aabb - abs_s*W_aabb) / det
    
    determinant = abs_c*abs_c - abs_s*abs_s

    if abs(determinant) < 1e-9:  # Cas ambigu (angle ~45°, ~135°, etc.)
        print(f"AVERTISSEMENT: Cas ambigu pour angle ~{angle_degrees} degrés. L'AABB ({W_aabb:.2f}x{H_aabb:.2f}) "
              f"ne permet pas de déterminer l'OBB unique.")
        # Solution de repli : reconstruire un carré OBB qui s'inscrit dans l'AABB.
        # w_obj + h_obj = W_aabb / abs_c (car W_aabb = H_aabb et abs_c = abs_s)
        # Note: cette approximation est souvent incorrecte si l'OBB original n'était pas un carré.
        if abs_c > 1e-9 :
            sum_dims = W_aabb / abs_c 
        else: # Si abs_c est aussi proche de zero (angle ~90 pour abs_c), alors abs_s est ~1
              # ce cas ne devrait pas arriver si determinant est proche de zero, sauf si W_aabb, H_aabb sont aussi zero
            sum_dims = W_aabb * 1.41421356 # Approximativement W_aabb * sqrt(2)
        
        reconstructed_width = sum_dims / 2.0
        reconstructed_height = sum_dims / 2.0
        print(f"Reconstruction approximative comme un carré OBB: largeur={reconstructed_width:.2f}, hauteur={reconstructed_height:.2f}")

    else:
        reconstructed_width = (abs_c * W_aabb - abs_s * H_aabb) / determinant
        reconstructed_height = (abs_c * H_aabb - abs_s * W_aabb) / determinant

    # S'assurer que les dimensions reconstruites sont positives
    if reconstructed_width < 0:
        # Cela peut arriver si le determinant est négatif et le numérateur positif.
        # Ex: angle 60 deg, det = -0.5.  (cW-sH) peut être positif.
        # Dans ce cas, les formules sont correctes, le signe négatif du déterminant s'inverse.
        # Cependant, pour éviter toute confusion, on prend la valeur absolue,
        # car les largeurs/hauteurs doivent être positives.
        # Une analyse plus approfondie pourrait être nécessaire si cela se produit souvent.
        print(f"Note: largeur reconstruite initialement négative ({reconstructed_width:.2f}), prise en valeur absolue.")
        reconstructed_width = abs(reconstructed_width)
    if reconstructed_height < 0:
        print(f"Note: hauteur reconstruite initialement négative ({reconstructed_height:.2f}), prise en valeur absolue.")
        reconstructed_height = abs(reconstructed_height)


    # print(f"bounding_box_to_polygon: AABB L={W_aabb:.2f}, H={H_aabb:.2f}, Angle={angle_degrees}")
    # print(f"bounding_box_to_polygon: OBB reconstruit - largeur={reconstructed_width:.2f}, hauteur={reconstructed_height:.2f}")

    return _calculate_obb_corners(center_x, center_y, reconstructed_width, reconstructed_height, angle_degrees)

def rescale_coords(polygon: List[tuple], ratios: tuple[float, float]) -> List[tuple]:
    """
    Rescale polygon coordinates based on given ratios.

    Args:
        polygon (List[tuple]): List of (x, y) tuples representing polygon points.
        ratios (tuple): Tuple containing (ratio_x, ratio_y) for rescaling.

    Returns:
        List[tuple]: Rescaled polygon coordinates.
    """
    ratio_x, ratio_y = ratios
    rescaled_polygon = [(int(x / ratio_x), int(y / ratio_y)) for (x, y) in polygon]
    return rescaled_polygon

def offset_polygon(polygon: List[tuple], offset: int) -> List[tuple]:
    """
    Offset polygon coordinates by a given value.

    Args:
        polygon (List[tuple]): List of (x, y) tuples representing polygon points.
        offset (int): Offset value to be added to both x and y coordinates.

    Returns:
        List[tuple]: Offset polygon coordinates.
    """
    offset_polygon = [(x + offset, y) for (x, y) in polygon]
    return offset_polygon

def postprocess_line(output : str, ratios: tuple[float, float] = (1.0, 1.0), offset: int = 0, char_perplexity: float = 0.0, line_perplexity: float = 0.0) -> dict:
    line = output.strip()
    # rsplit from the right with max 5 splits: handles text that contains tabs
    parts = line.rsplit('\t', 5)
    if len(parts) == 6:
        text, xmin, ymin, xmax, ymax, angle = parts
    else:
        # Try comma-separated
        res_values = []
        for value in line.split('\t'):
            res_values += value.split(",")
        if len(res_values) == 6:
            text, xmin, ymin, xmax, ymax, angle = res_values
        else:
            # Fallback: whitespace-separated, text may contain spaces or numbers
            m = re.match(r'^(.*)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?\d+)\s*$', line)
            if m:
                text, xmin, ymin, xmax, ymax, angle = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)
            else:
                raise ValueError(f"Format de ligne inattendu: '{output}'. Attendu: 'text\\txmin\\tymin\\txmax\\tymax\\tangle'.")
    xmin = float(xmin)
    ymin = float(ymin)
    xmax = float(xmax)
    ymax = float(ymax)
    angle = int(angle)
    
    polygon = bounding_box_to_polygon(xmin, ymin, xmax, ymax, angle)
    polygon = rescale_coords(polygon, ratios)
    polygon = offset_polygon(polygon, offset)
    return {
        "label": text,
        "points": polygon,
        "char_perplexity": round(char_perplexity, 3),
        "line_perplexity": round(line_perplexity, 3)
    }
def postprocess_output(output: str, ratios: tuple[float, float] = (1.0, 1.0), offset: int = 0) -> List[dict]:
    """
    Post-process the model output to extract text and polygon coordinates.
    Args:
        output (str): The raw output string from the model.
    Returns:
        List[dict]: A list of dictionaries containing 'text' and 'polygon' keys.
    """    
    results = []
    lines = output.strip().split('\n')
    for line in lines:
        if not "```" in line:
            try:
                processed_line = postprocess_line(line, ratios, offset)
                results.append(processed_line)
            except Exception as e:
                print(f"Erreur lors du traitement de la ligne: {line}. Détails de l'erreur: {e}", flush=True)
    return results

def add_metadata(lines: List[dict], metadata: dict) -> List[dict]:
    """
    Add metadata to each output dictionary.
    Args:
        outputs (List[dict]): List of output dictionaries.
        metadata (Dict): Metadata to be added.
    Returns:
        List[dict]: Updated list of output dictionaries with metadata.
    """
    output = {}
    output.update(metadata)
    output["shapes"] = lines
    return output