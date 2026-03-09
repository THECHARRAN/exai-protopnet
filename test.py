from inference.predictor import MRIEnsemblePredictor
from explanations.heatmap import generate_heatmap, visualize_heatmap

predictor = MRIEnsemblePredictor()

results,img,x,proto = predictor.predict(
    "dataset/test/meningioma/Te-aug-me_17.jpg"
)

for r in results:
    print(r)

heat = generate_heatmap(proto,x,"cpu")

visualize_heatmap(img,heat)