"""
app/core — núcleo del PRODUCTO: imagen → detección → análisis espacial →
narrativa egocéntrica → TTS.

Regla de dependencias (verificada por tests/architecture): app.core no importa
FastAPI, app.routes, app.catalog ni ningún módulo de evaluación o estudio.
Los módulos de dominio (detector, análisis espacial, narrativa, TTS) siguen en
app/services y app/utils con sus nombres de archivo originales; este paquete
contiene el orquestador único que usan /api/detect y el runner de evaluación.
"""
