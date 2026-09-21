# Conjunto de imágenes de evaluación Web3D

## Ubicación recomendada en el repositorio

Colocar el contenido de esta carpeta bajo:

`D:\University\T2\Programacion\vision-api-object-detection\evaluation\`

La estructura recomendada queda:

```text
evaluation/
├── images/
│   └── web3d/
│       ├── generated/
│       │   └── synthetic/
│       └── internet/
│           ├── sketchfab/
│           ├── polyhaven/
│           ├── justcreate3d/
│           └── kaykit/
└── metadata/
    ├── inventario_imagenes.csv
    └── fuentes_y_licencias.csv
```

## Importante

- Estas imágenes son un conjunto de evaluación y NO deben mezclarse con `test_images/`, que corresponde al conjunto experimental anterior del Capítulo 3.
- Tampoco deben colocarse en `dataset/` si ese directorio se utiliza para datos de entrenamiento/fine-tuning.
- La carpeta `generated/synthetic/` contiene imágenes sintéticas proporcionadas para la evaluación; no deben presentarse como imágenes descargadas de una fuente externa.
- Las imágenes procedentes de Internet deben conservar su fuente y licencia.
- En Sketchfab, la licencia se debe verificar para el asset específico; el hecho de que una imagen esté en Sketchfab no significa que tenga una licencia única aplicable a todo el sitio.
- Poly Haven declara que sus assets son CC0.
- KayKit Restaurant Bits declara licencia CC0.
- JustCreate3D usa una licencia propia para el pack; conservar la referencia y revisar sus condiciones.

## Ground truth

Este ZIP NO contiene todavía anotaciones de ground truth. Antes de calcular Precision, Recall o F1, cada imagen debe ser inspeccionada y anotada según los objetos realmente presentes.

## Nomenclatura

- `W3D-G-*`: imagen sintética generada.
- `W3D-I-SKF-*`: imagen de Sketchfab.
- `W3D-I-PHV-*`: imagen de Poly Haven.
- `W3D-I-JC3D-*`: imagen de JustCreate3D.
- `W3D-I-KYK-*`: imagen de KayKit.

La numeración es un identificador técnico y no implica orden de calidad ni prioridad experimental.
