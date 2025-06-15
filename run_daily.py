import os

# 1. Colectează date pentru ultimele 300 zile
print("\n📥 Collecting latest CSVs...")
os.system("python collect_csvs.py 300")

# 2. Antrenează toate modelele Transformer
print("\n🧠 Training all transformer models...")
os.system("python train_all_transformers.py")

# 3. Rulează predicțiile pentru toate modelele
print("\n🔮 Running predictions for all symbols...")
os.system("python predict_all_transformers.py")

print("\n✅ Daily pipeline finished.")
