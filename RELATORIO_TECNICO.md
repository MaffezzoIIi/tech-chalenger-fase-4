# Relatório Técnico — Tech Challenge (Fase 4)
**Projeto: Sistema de Monitoramento Multimodal Preventivo Hospitalar (100% Open Source)**

---

## 1. Descrição do Fluxo Multimodal

O sistema integra quatro pipelines paralelos que convergem para um **Motor de Fusão Multimodal**:
1. **Pipeline de Vídeo**: Captura frames de sessões de fisioterapia ou monitoramento de leito, extrai keypoints posturais (**OpenPose / YOLOv8-Pose**) e bounding boxes (**YOLOv8**), identificando inclinações anômalas de tronco ($> 20^\circ$), assimetria motora e invasão/evasão de áreas críticas (ROIs).
2. **Pipeline de Áudio & NLP**: Extrai biomarcadores acústicos via **Librosa** (*Pitch $F_0$*, *Jitter*, *Shimmer*, *Pause Ratio*), transcreve a fala localmente com **OpenAI Whisper (`faster-whisper`)** e identifica termos clínicos críticos e sentimento negativo.
3. **Pipeline de Anomalias**: Aplica **Isolation Forest**, **Z-Score Móvel (EWMA)** e **NEWS2** sobre séries temporais reais do **PhysioNet BIDMC** (`HR`, `SpO2`, `RESP`), além de auditar prescrições reais do **MIMIC-III Demo** e telemetria de movimentação no leito.
4. **Fusão & Alerta**: Combina os escores normalizados $[0, 1]$ com fator de sinergia clínica ($\Phi_{\text{sinergia}}$) e grava alertas em tempo real em `logs/multimodal_alerts.jsonl` e SQLite.

---

## 2. Modelos Aplicados em Cada Tipo de Dado

| Modalidade | Modelos / Técnicas Aplicadas | Objetivo Clínico |
| :--- | :--- | :--- |
| **Vídeo** | `YOLOv8` (`yolov8n.pt`) + `YOLOv8-Pose` / Topologia `OpenPose` | Detecção de objetos/ROIs e desvios posturais |
| **Áudio** | `Librosa` (YIN, RMS, Split) + `faster-whisper` (`tiny`) | Fadiga vocal, disartria, dispneia e transcrição |
| **Texto (NLP)** | Motor NLP Clínico Open Source (Léxico Médico + Polaridade) | Termos críticos e sofrimento agudo |
| **Sinais Vitais** | `Isolation Forest` + `Z-Score EWMA` + `NEWS2` | Deterioração cardiorrespiratória precoce |
| **Prescrições** | Análise de série temporal de doses + regras de alto risco | Escalonamento abrupto / erros de dosagem |

---

## 3. Justificativa Arquitetural: Open Source Local vs. Cloud Proprietária

A adoção de motores **100% Open Source executados localmente (On-Premise)** garante:
- **Conformidade LGPD (Art. 11)** e sigilo médico (CFM), pois nenhum vídeo, áudio ou prontuário deixa a rede hospitalar;
- **Custo zero de licenciamento** e independência de conectividade externa em ambientes críticos de UTI.

---

## 4. Exemplos de Anomalias Detectadas

- **PAC-4001 (Neurologia / Fisioterapia)**: Assimetria de membros superiores e inclinação de tronco no vídeo + *Jitter* e *Pause Ratio* elevados no áudio + termos críticos (*"dor no peito"*, *"tontura"*, *"fraqueza"*) $\rightarrow$ Alerta multimodal integrado.
- **PAC-4002 (UTI Pós-Cirúrgica)**: Evasão da zona segura do leito no vídeo + agitação psicomotora + escalonamento abrupto de *Noradrenalina* nas prescrições $\rightarrow$ Alerta de alto risco para a equipe médica.
