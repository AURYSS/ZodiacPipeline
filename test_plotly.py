import plotly.express as px
import pandas as pd

try:
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    fig = px.scatter(df, x='x', y='y')
    img_bytes = fig.to_image(format="png")
    print("Success, bytes:", len(img_bytes))
except Exception as e:
    print("ERROR:", str(e))
