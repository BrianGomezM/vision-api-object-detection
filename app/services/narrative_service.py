# app/services/narrative_service.py
"""
Narrativa egocéntrica avanzada - Fallback rápido sin LLM
"""
from collections import defaultdict

def generate_narrative(spatial_data):
    """Genera instrucciones claras usando la cuadrícula 3x3."""
    if not spatial_data:
        return "No se detectaron objetos. Espacio despejado."
    
    zonas = defaultdict(list)
    for obj in spatial_data:
        zonas[obj["position"]].append(obj["label"])
    
    frases = []
    
    # Prioridad 1: Objetos muy cerca (peligro de choque)
    if zonas.get("center_near"):
        frases.append(f"¡Cuidado! Tienes {', '.join(zonas['center_near'][:2])} justo frente a ti.")
    
    # Prioridad 2: Objetos cerca izquierda/derecha
    if zonas.get("left_near"):
        frases.append("Obstáculo a tu izquierda inmediata.")
    if zonas.get("right_near"):
        frases.append("Obstáculo a tu derecha inmediata.")
    
    # Prioridad 3: Objetos a distancia media
    if zonas.get("center"):
        frases.append(f"Frente a ti: {', '.join(zonas['center'][:3])}.")
    
    # Izquierda (left_mid)
    if zonas.get("left_mid"):
        frases.append(f"A tu izquierda: {', '.join(zonas['left_mid'][:2])}.")
    elif zonas.get("left_far") and not zonas.get("left_near"):
        frases.append("A tu izquierda hay espacio.")
    
    # Derecha (right_mid)
    if zonas.get("right_mid"):
        frases.append(f"A tu derecha: {', '.join(zonas['right_mid'][:2])}.")
    elif zonas.get("right_far") and not zonas.get("right_near"):
        frases.append("A tu derecha hay espacio.")
    
    # Prioridad 4: Objetos al fondo
    if zonas.get("center_far"):
        frases.append(f"Al fondo: {', '.join(zonas['center_far'][:2])}.")
    
    # Prioridad 5: Sugerencia de avance
    if not zonas.get("center_near"):
        frases.append("Puedes avanzar con seguridad.")
    
    if not frases:
        return "Espacio despejado. Puedes moverte en cualquier dirección."
    
    return " ".join(frases)