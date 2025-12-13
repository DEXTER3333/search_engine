"""
app.py - Lance le serveur API et affiche frontend.html
Version simplifiée qui réutilise le code existant
"""

import sys
import os
import threading
import time
import webbrowser

# Ajouter le chemin du dossier au sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🔄 Initialisation du moteur ML/DL...")
print("=" * 70)

# Vérifier l'existence des fichiers
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "emb.db")
frontend_path = os.path.join(script_dir, "frontend.html")

print(f"📁 Répertoire: {script_dir}")
print(f"📊 DB existe: {os.path.exists(db_path)}")
print(f"📄 Frontend existe: {os.path.exists(frontend_path)}")

if os.path.exists(db_path):
    db_size = os.path.getsize(db_path) / (1024*1024)
    print(f"📊 Taille DB: {db_size:.1f} MB")
else:
    print("\n⚠️  ATTENTION: emb.db manquant!")
    print("   Lancez: python3 main_optimized.py crawl\n")

if not os.path.exists(frontend_path):
    print("\n⚠️  ATTENTION: frontend.html manquant!")
    print("   Assurez-vous que frontend.html est dans le même dossier\n")

print("=" * 70)

def start_api_server():
    """Démarre le serveur FastAPI"""
    try:
        import uvicorn
        from api_server import app
        
        print("\n🌐 Démarrage du serveur API...")
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=8000, 
            log_level="info"
        )
    except Exception as e:
        print(f"\n❌ Erreur serveur API: {e}")
        import traceback
        traceback.print_exc()

# Lancer le serveur dans un thread séparé
print("\n🚀 Lancement du serveur en arrière-plan...")
api_thread = threading.Thread(target=start_api_server, daemon=True)
api_thread.start()

# Attendre que le serveur démarre
print("⏳ Attente du démarrage (5 secondes)...")
time.sleep(5)

print("\n" + "=" * 70)
print("✅ ✅ ✅  SERVEUR EN LIGNE  ✅ ✅ ✅")
print("=" * 70)
print("\n🌍 Interface web disponible sur:")
print("   http://127.0.0.1:8000")
print("\n📖 Documentation API:")
print("   http://127.0.0.1:8000/docs")
print("\n📊 Statistiques:")
print("   http://127.0.0.1:8000/stats")
print("\n⚡ Benchmark:")
print("   http://127.0.0.1:8000/benchmark")
print("\n" + "=" * 70)
print("\n💡 Le serveur utilise:")
print("   - frontend.html pour l'interface")
print("   - api_server.py pour le backend")
print("   - indexer_optimized.py pour la recherche ML/DL")
print("\n⚠️  Gardez cette fenêtre ouverte!")
print("   Appuyez sur Ctrl+C pour arrêter le serveur")
print("=" * 70 + "\n")

# Ouvrir le navigateur automatiquement
try:
    print("🌐 Ouverture du navigateur...")
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:8000")
except:
    pass

# Maintenir le script actif
try:
    print("🔄 Serveur actif... (Ctrl+C pour arrêter)\n")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\n🛑 Arrêt du serveur...")
    print("✅ Au revoir!")