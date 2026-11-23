"""Gerenciador de banco de dados SQLite para monitoramento multimodal hospitalar."""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List

from src.config import DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Cria as tabelas do banco hospitalar multimodal e popula pacientes iniciais."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name_anon TEXT NOT NULL,
            age INTEGER NOT NULL,
            ward TEXT NOT NULL,
            diagnosis_hypothesis TEXT NOT NULL,
            allergies TEXT NOT NULL,
            vitals_file TEXT NOT NULL,
            video_key TEXT NOT NULL,
            audio_key TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            severity TEXT NOT NULL,
            risk_score REAL NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    patients = [
        (
            "PAC-4001",
            "PACIENTE_A_ANON",
            68,
            "Neurologia / Fisioterapia",
            "Suspeita de AVC Isquêmico Agudo / Déficit Motor",
            "Dipirona",
            "bidmc_01_Numerics.csv",
            "fisioterapia",
            "consulta_fadiga_disartria",
        ),
        (
            "PAC-4002",
            "PACIENTE_B_ANON",
            72,
            "UTI Pós-Cirúrgica",
            "Descompensação Cardiorrespiratória / Sepse",
            "Penicilina",
            "bidmc_02_Numerics.csv",
            "leito_cirurgia",
            "consulta_fadiga_disartria",
        ),
        (
            "PAC-4003",
            "PACIENTE_C_ANON",
            54,
            "Enfermaria Clínica",
            "Monitoramento Pós-Operatório Estável",
            "Nenhuma declarada",
            "bidmc_03_Numerics.csv",
            "fisioterapia",
            "consulta_normal",
        ),
    ]

    cur.executemany(
        """
        INSERT OR REPLACE INTO patients
        (patient_id, name_anon, age, ward, diagnosis_hypothesis, allergies, vitals_file, video_key, audio_key)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        patients,
    )
    conn.commit()
    conn.close()


def list_patients() -> List[Dict[str, Any]]:
    init_database()
    conn = get_connection()
    rows = conn.execute("SELECT * FROM patients ORDER BY patient_id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient(patient_id: str) -> Dict[str, Any] | None:
    init_database()
    conn = get_connection()
    row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def log_alert_to_db(patient_id: str, severity: str, risk_score: float, summary: str, created_at: str) -> None:
    init_database()
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO alerts_history (patient_id, severity, risk_score, summary, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (patient_id, severity, risk_score, summary, created_at),
    )
    conn.commit()
    conn.close()
