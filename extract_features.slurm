#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --partition=debug
#SBATCH --output=logs/feature_extraction_%j.out
#SBATCH --error=logs/feature_extraction_%j.err
#SBATCH --job-name=stressid_extract_group14

# 🛸 Project Context: StressID Multimodal Pipeline (Group 14)
# Target Host: USC CARC (Discovery/Antigravity Cluster)
# Project Root: /project2/msoleyma_1026/group_14/

cd /project2/msoleyma_1026/group_14/multimodal-stress-level-measurement

module load python/3.10
source /project2/msoleyma_1026/group_14/your-env-name/bin/activate

echo "[START] StressID Dataset Feature Extraction (No-JPG Rule)"

# Run Face Extraction
echo "-> Extracting Face Landmarks..."
python feature_extraction/face_feature_extractor.py --split train
python feature_extraction/face_feature_extractor.py --split test

# Run Gesture Extraction
echo "-> Extracting Gesture Landmarks..."
python feature_extraction/gesture_feature_extractor.py --split train
python feature_extraction/gesture_feature_extractor.py --split test

# Run Audio Extraction
echo "-> Extracting Mel Spectrograms..."
python feature_extraction/audio_processor.py --split train
python feature_extraction/audio_processor.py --split test

echo "[END] Feature Extraction Pipeline Complete."
