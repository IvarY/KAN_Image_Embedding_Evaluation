# KAN_Image_Embedding_Evaluation

Aggregated results (KNN-precision, sillhouette scores) are saved in ```./results/``` as .csv files.

Training history, embeddings, model checkpoints and t-SNE plots are saved in respective subfolders: ```./results/model_name/```.



Tested models rely on the following KAN implementations:
- Efficient KAN: https://github.com/Blealtan/efficient-kan
- Wavelet KAN: https://github.com/zavareh1/Wav-KAN

To run the dependant models, create folder ```./kan_repos/``` and clone the repositories there:
```bash
mkdir kan_repos
cd kan_repos
git clone https://github.com/zavareh1/Wav-KAN.git
git clone https://github.com/Blealtan/efficient-kan.git
