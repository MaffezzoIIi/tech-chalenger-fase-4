# 🏥 Monitoramento Multimodal Preventivo Hospitalar — Tech Challenge (Fase 4)

> Projeto desenvolvido para o **Tech Challenge - Fase 4** da Pós-Graduação em Inteligência Artificial (Pós Tech).  
> **Arquitetura 100% Open Source & Local (On-Premise)** operando sobre **dados reais públicos (PhysioNet BIDMC & MIMIC-III Demo)**.

---

## 🎯 Entregas Técnicas Atendidas

1. **Análise de Vídeo Clínico (`OpenPose` + `YOLOv8`)**:
   - Estimação postural de keypoints corporais e cálculo de ângulos biomecânicos (`src/video_analysis/openpose_analyzer.py`);
   - Detecção de objetos, pacientes e monitoramento de Áreas Críticas (ROIs) com **YOLOv8** (`src/video_analysis/yolo_detector.py`);
   - Geração automática de relatórios de desvios e frames anotados (`src/video_analysis/video_reporter.py`).
2. **Análise de Áudio e NLP Clínico (`Librosa` + `OpenAI Whisper` + `NLP`)**:
   - Extração de biomarcadores vocais objetivos ($F_0$, *Jitter*, *Shimmer*, *Pause Ratio*) para detecção de **fadiga vocal, disartria e dispneia** (`src/audio_analysis/vocal_biomarkers.py`);
   - Transcrição local em português via **faster-whisper** (`src/audio_analysis/speech_to_text.py`);
   - Análise de sentimento e extração de termos clínicos críticos (`src/audio_analysis/text_analytics.py`).
3. **Detecção de Anomalias em Tempo Real (`Isolation Forest` + `EWMA Z-Score` + `NEWS2`)**:
   - Séries temporais reais de sinais vitais do **PhysioNet BIDMC** (`src/anomaly_detection/vital_signs_detector.py`);
   - Evolução de prescrições reais do **PhysioNet MIMIC-III Demo** (`src/anomaly_detection/prescription_detector.py`);
   - Padrões de movimentação no leito: agitação, risco de queda e imobilidade (`src/anomaly_detection/movement_detector.py`).
4. **Fusão Multimodal e Alertas Médicos em Tempo Real**:
   - Correlação cruzada entre modalidades com bônus de sinergia clínica (`src/multimodal_fusion/fusion_engine.py`);
   - Emissão de alertas estruturados com trilha de auditoria JSONL e conformidade CFM/LGPD (`src/multimodal_fusion/alert_dispatcher.py`).

---

## 🚀 Como Executar

### 1. Instalar Dependências
```powershell
py -m pip install -r requirements.txt
```

### 2. Baixar Dados Reais (PhysioNet) e Preparar Datasets
```powershell
py -m src.data_prep.real_data_downloader
```

### 3. Abrir o Dashboard Visual Interativo (Recomendado para o Vídeo de 15 min)
```powershell
py -m streamlit run src/dashboard.py
```

### 4. Executar via CLI (Terminal)
```powershell
py -m src.cli --run-all
```

### 5. Rodar Suíte de Testes Automatizados
```powershell
py -m pytest tests/ -v
```

---

## 🎬 Roteiro Sugerido para o Vídeo de Demonstração (até 15 min)
1. **Introdução (2 min)**: Contexto do hospital, evolução da Fase 3 para a Fase 4 e justificativa da arquitetura **100% Open Source Local** (privacidade LGPD e custo zero).
2. **Dados Reais & Arquitetura (2 min)**: Mostrar os arquivos baixados do **PhysioNet** em `data/raw/`.
3. **Demo ao Vivo no Dashboard Streamlit (8 min)**:
   - Aba **Fusão Multimodal**: Selecionar `PAC-4001` e `PAC-4002`, mostrar o escore integrado e disparar o alerta em tempo real.
   - Aba **Vídeo**: Mostrar o frame anotado com **OpenPose/YOLOv8** e a tabela de desvios.
   - Aba **Áudio & NLP**: Tocar o áudio, mostrar Jitter/Shimmer/Pause Ratio, transcrição **Whisper** e termos críticos.
   - Aba **Anomalias**: Mostrar os gráficos de sinais vitais do **PhysioNet** e prescrições do **MIMIC-III**.
4. **Testes e Auditoria (3 min)**: Rodar `py -m pytest tests/ -v` e mostrar o log `logs/multimodal_alerts.jsonl`.
