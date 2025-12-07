from huggingface_hub import HfApi, login

# Se connecter (vous aurez besoin d'un token d'accès)
login(token=".")  # Ou utilisez login(token="votre_token")

# Initialiser l'API
api = HfApi()

api.create_repo(
    repo_id="molecular-captioning-model",
    repo_type="model",
    private=False  # Mettez True si vous voulez un repo privé
)
# Upload d'un fichier
api.upload_file(
    path_or_fileobj="saved_model/model_gpt2_epoch_20.pth",
    path_in_repo="modele_epoch20.pth",  # nom du fichier dans le repo
    repo_id="GAD-cell/molecular-captioning-model",
    repo_type="model"
)