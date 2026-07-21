from modules.database import obtener_todas_las_encuestas
print("Retrieving surveys...")
df = obtener_todas_las_encuestas()
print("Retrieved successfully. Shape:", df.shape)
