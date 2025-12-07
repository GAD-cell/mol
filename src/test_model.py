import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

# Vos imports personnalisés (ajustez les chemins si nécessaire)
from src.model.test_t5 import MoLCABackbone_T5
from src.data.data_process import PreprocessedGraphDataset, collate_fn
# Assurez-vous d'importer MolecularCaptionEvaluator (probablement de src.utils)
from src.utils import MolecularCaptionEvaluator 

def run_inference():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "src/saved_model/best_model_gpt2.pth"
    val_data_path = "src/data/train_graphs_smiles.pkl"
    
    print(f"Chargement du modèle sur {device}...")

    # 2. Reconstruire l'architecture EXACTEMENT comme à l'entraînement
    model = MoLCABackbone_T5(
        model_name="GT4SD/multitask-text-and-chemistry-t5-base-standard",
        graph_hidden_dim=300,
        freeze_encoder=True, 
        freeze_decoder=True  
    )

    # 3. Charger les poids (State Dict)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) 
    
    model.to(device)
    model.eval() 
    print("Modèle chargé avec succès !")

    # --- NOUVEAU : Initialisation de l'évaluateur ---
    evaluator = MolecularCaptionEvaluator(device=device)
    # ------------------------------------------------

    # 4. Charger quelques données de validation
    print("Chargement des données...")
    val_dataset = PreprocessedGraphDataset(val_data_path, encode_feat=True)
    
    # On prend 4 exemples pour tester
    subset = Subset(val_dataset, indices=[0, 1, 2, 3]) 
    loader = DataLoader(subset, batch_size=4, collate_fn=collate_fn, shuffle=False)

    # 5. Inférence
    print("\n--- DÉBUT DES PRÉDICTIONS ---")
    
    batch = next(iter(loader))
    batch_graph, batch_smiles, batch_descriptions = batch
    batch_graph = batch_graph.to(device)

    # Stockage des légendes pour l'évaluation globale
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        generated_captions = model.generate(
            mol_data=batch_graph, 
            smiles_text=batch_smiles, 
            max_length=512, 
            num_beams=5
        )
        
        all_predictions.extend(generated_captions)
        all_references.extend(batch_descriptions)

    # 6. Affichage des résultats individuels
    for i in range(len(batch_smiles)):
        print(f"\n🧪 Molécule {i+1} (SMILES): {batch_smiles[i][:50]}...")
        print(f"✅ Vérité Terrain : {all_references[i]}")
        print(f"🤖 Prédiction T5  : {all_predictions[i]}")
        print("-" * 80)
        
    # --- NOUVEAU : Calcul et Affichage des Scores ---
    eval_results = evaluator.evaluate_batch(all_predictions, all_references)
    
    print("\n\n=== SCORES DÉTAILLÉS (sur ces 4 exemples) ===")
    
    # Le score BLEU F1 que vous cherchez est souvent BLEU-4, 
    # mais regardons le détail des métriques fournies par l'évaluateur :
    
    if 'bleu4_f1_mean' in eval_results:
        print(f"🔹 BLEU (n-grammes exacts, F1) : {eval_results['bleu4_f1_mean']:.4f}")
    if 'bertscore_f1_mean' in eval_results:
        print(f"🔹 BERTScore (Sémantique, F1) : {eval_results['bertscore_f1_mean']:.4f}")
    
    print(f"⭐ Score Composite (Moyenne) : {eval_results['composite_score']:.4f}")
    print("==============================================")


if __name__ == "__main__":
    run_inference()