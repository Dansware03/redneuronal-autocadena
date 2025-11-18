import subprocess
import os

# Ruta al ejecutable ffmpeg (si está en el PATH, no necesitas ruta completa)
ffmpeg_cmd = "ffmpeg"

# Ruta del video fuente
input_video = r"C:\Dansware_dev\extraccionCadena\normal\cadena2.mp4"

# Carpeta de salida
output_dir = r"C:\Dansware_dev\redautocadena\data\nuevo"
os.makedirs(output_dir, exist_ok=True)

# Comando FFmpeg: extraer 1 frame cada 5 frames
output_pattern = os.path.join(output_dir, "vtv_%04d.jpg")
cmd = [
    ffmpeg_cmd,
    "-i", input_video,
    "-vf", "select='not(mod(n,5))',setpts=N/FRAME_RATE/TB",
    "-q:v", "2",
    output_pattern
]

# Ejecutar el comando
print("Extrayendo 1 frame cada 5 frames con FFmpeg...")
subprocess.run(cmd, check=True)
print("Frames guardados en:", output_dir)