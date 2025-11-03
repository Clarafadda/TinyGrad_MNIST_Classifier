#!/usr/bin/env python3
"""
Script d'exploration avancée des hyperparamètres
Teste différents types de paramètres : LR, BS, architecture, etc.
Usage: python advanced_hyperparameter_explorer.py --model mlp
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

# ==================== MLP CONFIGURATIONS ====================
MLP_CONFIGS = [
    # Config 1: Baseline
    {
        "LR": 0.001,
        "BS": 128,
        "STEPS": 100,
        "description": "Baseline - Adam with standard params",
        "category": "baseline"
    },

    # Config 2-3: Learning Rate variation
    {
        "LR": 0.003,
        "BS": 128,
        "STEPS": 100,
        "description": "Higher LR - faster convergence test",
        "category": "learning_rate"
    },
    {
        "LR": 0.0005,
        "BS": 128,
        "STEPS": 100,
        "description": "Lower LR - stability test",
        "category": "learning_rate"
    },

    # Config 4-5: Batch Size variation
    {
        "LR": 0.001,
        "BS": 256,
        "STEPS": 100,
        "description": "Larger batch - smoother gradients",
        "category": "batch_size"
    },
    {
        "LR": 0.001,
        "BS": 64,
        "STEPS": 100,
        "description": "Smaller batch - more updates per epoch",
        "category": "batch_size"
    },

    # Config 6: Training duration
    {
        "LR": 0.001,
        "BS": 128,
        "STEPS": 1000,
        "description": "Extended training - convergence test",
        "category": "training_duration"
    },

    # Config 7: Optimized combo
    {
        "LR": 0.002,
        "BS": 256,
        "STEPS": 500,
        "description": "Aggressive - high LR + large batch",
        "category": "combo"
    },

    # Config 8: Fine-tuning
    {
        "LR": 0.0008,
        "BS": 128,
        "STEPS": 500,
        "description": "Conservative - balanced approach",
        "category": "combo"
    },

]

# ==================== CNN CONFIGURATIONS ====================
CNN_CONFIGS = [
    # Config 1: Baseline
    {
        "LR": 0.001,
        "BS": 128,
        "STEPS": 100,
        "description": "Baseline CNN - standard params",
        "category": "baseline"
    },

    # Config 2-3: Learning Rate (CNNs often need lower LR)
    {
        "LR": 0.0005,
        "BS": 128,
        "STEPS": 100,
        "description": "Lower LR - CNNs converge slower",
        "category": "learning_rate"
    },
    {
        "LR": 0.002,
        "BS": 128,
        "STEPS": 100,
        "description": "Higher LR - risk of instability",
        "category": "learning_rate"
    },

    # Config 4-5: Batch Size (CNNs are memory-intensive)
    {
        "LR": 0.001,
        "BS": 64,
        "STEPS": 100,
        "description": "Smaller batch - better for CNNs",
        "category": "batch_size"
    },
    {
        "LR": 0.001,
        "BS": 256,
        "STEPS": 100,
        "description": "Larger batch - faster training",
        "category": "batch_size"
    },

    # Config 6: Extended training
    {
        "LR": 0.001,
        "BS": 128,
        "STEPS": 1000,
        "description": "More steps - better convergence",
        "category": "training_duration"
    },

    # Config 7: Mini-batch + more steps
    {
        "LR": 0.001,
        "BS": 32,
        "STEPS": 500,
        "description": "Mini-batch - fine-grained updates",
        "category": "combo"
    },

    # Config 8: Optimal combo
    {
        "LR": 0.0008,
        "BS": 64,
        "STEPS": 500,
        "description": "Balanced - optimal CNN settings",
        "category": "combo"
    },

]


# ==================== HELPER FUNCTIONS ====================

def run_experiment(model_type, config, config_num):
    """Lance un entraînement avec une configuration donnée"""
    print(f"\n{'=' * 70}")
    print(f"Experiment {config_num}/{len(CNN_CONFIGS if model_type == 'convnet' else MLP_CONFIGS)}")
    print(f"Model: {model_type.upper()}")
    print(f"Category: {config.get('category', 'general')}")
    print(f"Description: {config['description']}")
    print(f"Parameters:")
    for key, val in config.items():
        if key not in ['description', 'category']:
            print(f"   - {key}: {val}")
    print(f"{'=' * 70}\n")

    # Prépare les variables d'environnement
    env = os.environ.copy()
    #env['JIT'] = '1'  # Always enable JIT

    for key, val in config.items():
        if key not in ['description', 'category']:
            env[key.upper()] = str(val)

    # Nom du script à exécuter
    script_name = f"mnist_{model_type}.py"

    # Lance l'entraînement
    start_time = datetime.now()
    try:
        result = subprocess.run(
            ['python', script_name],
            env=env,
            capture_output=True,
            text=True,
            timeout=900  # 15 minutes max
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Affiche la sortie
        print(result.stdout)
        if result.stderr:
            print("⚠️  STDERR:", result.stderr)

        # Extrait l'accuracy de la sortie
        accuracy = extract_accuracy(result.stdout)

        return {
            "config_num": config_num,
            "description": config['description'],
            "category": config.get('category', 'general'),
            "params": {k: v for k, v in config.items() if k not in ['description', 'category']},
            "accuracy": accuracy,
            "training_time": round(duration, 2),
            "timestamp": datetime.now().isoformat()
        }

    except subprocess.TimeoutExpired:
        print("⚠️  TIMEOUT: L'entraînement a dépassé 15 minutes")
        return {
            "config_num": config_num,
            "description": config['description'],
            "category": config.get('category', 'general'),
            "params": {k: v for k, v in config.items() if k not in ['description', 'category']},
            "accuracy": None,
            "error": "Timeout",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"ERREUR: {str(e)}")
        return {
            "config_num": config_num,
            "description": config['description'],
            "category": config.get('category', 'general'),
            "params": {k: v for k, v in config.items() if k not in ['description', 'category']},
            "accuracy": None,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def extract_accuracy(output):
    """Extrait l'accuracy du test depuis la sortie standard"""
    import re

    # Cherche dans les dernières lignes (c'est là qu'est le résultat final)
    lines = output.split('\n')

    # Pattern 1: "Best accuracy: XX.XX%"
    for line in reversed(lines):
        match = re.search(r'best accuracy:\s*(\d+\.?\d*)%?', line.lower())
        if match:
            acc = float(match.group(1))
            if acc <= 1.0:
                acc *= 100
            return round(acc, 2)

    # Pattern 2: Dans la barre de progression "accuracy: XX.XX%"
    for line in reversed(lines):
        match = re.search(r'accuracy:\s*(\d+\.?\d*)%', line.lower())
        if match:
            acc = float(match.group(1))
            if acc <= 1.0:
                acc *= 100
            return round(acc, 2)

    # Pattern 3: "Training complete! Best accuracy: XX.XX%"
    for line in reversed(lines):
        match = re.search(r'training complete.*?(\d+\.?\d*)%', line.lower())
        if match:
            acc = float(match.group(1))
            if acc <= 1.0:
                acc *= 100
            return round(acc, 2)

    return None


def save_results(model_type, results):
    """Sauvegarde les résultats en JSON"""
    output_dir = Path("exploration_results")
    output_dir.mkdir(exist_ok=True)

    filename = output_dir / f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvegardés dans {filename}")
    return filename


def print_summary(results, model_type):
    """Affiche un résumé des résultats avec analyse"""
    print("\n" + "=" * 70)
    print(f"SUMMARY: {model_type.upper()} HYPERPARAMETER EXPLORATION")
    print("=" * 70)

    # Groupe par catégorie
    by_category = {}
    for r in results:
        cat = r.get('category', 'general')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    # Affiche par catégorie
    for category, configs in by_category.items():
        print(f"\n🔹 {category.upper().replace('_', ' ')}")
        for r in configs:
            acc_str = f"{r['accuracy']:.2f}%" if r['accuracy'] else "N/A"
            time_str = f"({r.get('training_time', '?')}s)" if 'training_time' in r else ""
            print(f"Config {r['config_num']}: {acc_str} {time_str}")
            print(f"     {r['description']}")

    # Meilleure config
    valid_results = [r for r in results if r['accuracy']]
    if valid_results:
        best = max(valid_results, key=lambda x: x['accuracy'])
        print(f"\n BEST CONFIGURATION:")
        print(f"   Config {best['config_num']}: {best['accuracy']:.2f}%")
        print(f"   {best['description']}")
        print(f"   Parameters: {best['params']}")

        # Target check
        target = 95 if model_type == 'mlp' else 98
        if best['accuracy'] >= target:
            print(f"TARGET REACHED! (>= {target}%)")
        else:
            print(f"Below target ({target}%). Consider more training or tuning.")

    print("\n" + "=" * 70)


def generate_markdown_report(mlp_results=None, cnn_results=None):
    """Génère un rapport markdown automatique"""
    output_file = Path("HYPERPARAMETERS.md")

    with open(output_file, 'w') as f:
        f.write("# Hyperparameter Exploration Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        if mlp_results:
            f.write("## 🧠 MLP (Multi-Layer Perceptron)\n\n")
            write_model_section(f, mlp_results, "MLP", 95)

        if cnn_results:
            f.write("\n---\n\n")
            f.write("## 🎯 CNN (Convolutional Neural Network)\n\n")
            write_model_section(f, cnn_results, "CNN", 98)

    print(f"\n📄 Markdown report generated: {output_file}")


def write_model_section(f, results, model_name, target):
    """Écrit une section du rapport markdown"""
    f.write(f"### Configurations Tested\n\n")
    f.write("| # | Category | Description | LR | BS | Steps | Accuracy | Time |\n")
    f.write("|---|----------|-------------|----|----|-------|----------|------|\n")

    for r in results:
        params = r['params']
        acc = f"{r['accuracy']:.2f}%" if r['accuracy'] else "Failed"
        time = f"{r.get('training_time', '?')}s"
        f.write(f"| {r['config_num']} | {r['category']} | {r['description']} | "
                f"{params.get('LR', '-')} | {params.get('BS', '-')} | {params.get('STEPS', '-')} | "
                f"{acc} | {time} |\n")

    # Best config
    valid = [r for r in results if r['accuracy']]
    if valid:
        best = max(valid, key=lambda x: x['accuracy'])
        f.write(f"\n### 🏆 Best Configuration\n\n")
        f.write(f"- **Accuracy:** {best['accuracy']:.2f}%\n")
        f.write(f"- **Description:** {best['description']}\n")
        f.write(f"- **Parameters:**\n")
        for k, v in best['params'].items():
            f.write(f"  - `{k}`: {v}\n")

        if best['accuracy'] >= target:
            f.write(f"\n✅ **Target reached!** (>= {target}%)\n")
        else:
            f.write(f"\n⚠️ **Below target** ({target}%). Consider further optimization.\n")


# ==================== MAIN ====================

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ['mlp', 'convnet', 'both']:
        print("Usage: python advanced_hyperparameter_explorer.py [mlp|convnet|both]")
        sys.exit(1)

    mode = sys.argv[1]

    mlp_results = None
    cnn_results = None

    if mode in ['mlp', 'both']:
        print(f"\n🚀 Starting MLP exploration ({len(MLP_CONFIGS)} configurations)...\n")
        mlp_results = []
        for i, config in enumerate(MLP_CONFIGS, 1):
            result = run_experiment('mlp', config, i)
            mlp_results.append(result)

        print_summary(mlp_results, 'mlp')
        save_results('mlp', mlp_results)

    if mode in ['convnet', 'both']:
        print(f"\n🚀 Starting CNN exploration ({len(CNN_CONFIGS)} configurations)...\n")
        cnn_results = []
        for i, config in enumerate(CNN_CONFIGS, 1):
            result = run_experiment('convnet', config, i)
            cnn_results.append(result)

        print_summary(cnn_results, 'convnet')
        save_results('convnet', cnn_results)

    # Generate markdown report
    if mode == 'both':
        generate_markdown_report(mlp_results, cnn_results)
    elif mode == 'mlp':
        generate_markdown_report(mlp_results=mlp_results)
    else:
        generate_markdown_report(cnn_results=cnn_results)

    print("\n✨ Exploration complete!")


if __name__ == "__main__":
    main()