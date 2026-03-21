"""
document_parser.py — Parse machine maintenance documents into text chunks.

Supports:
  - Plain text (.txt) files
  - Markdown (.md) files
  - In-memory built-in knowledge base (no files needed to get started)

The parser splits documents into overlapping chunks suitable for embedding.

Usage:
    from vector_pipeline.document_parser import DocumentParser
    parser = DocumentParser()
    chunks = parser.parse_builtin_knowledge()
    # or
    chunks = parser.parse_file("docs/sealing_unit_manual.txt", fault_type="machine_fault")
"""

import os
import re
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Built-in Knowledge Base
# These are concise maintenance docs for the tea bag packing machine.
# In production, replace or extend with real PDF/text manuals.
# ─────────────────────────────────────────────────────────────────────────────
BUILTIN_KNOWLEDGE: List[Dict[str, Any]] = [
    # ── Machine Fault Docs ────────────────────────────────────────────────
    {
        "doc_id": "builtin/machine_fault_motor",
        "title": "Motor Overload & Mechanical Fault — Diagnosis & Repair",
        "fault_type": "machine_fault",
        "sensor": "motor_current",
        "content": """
Motor Overload Fault — Tea Bag Packing Machine

Symptoms:
- Motor current exceeds 6.0 A (normal range: 3–6 A)
- Vibration increases above 2.0 mm/s
- Production speed drops below 130 bpm
- Machine may stop automatically to protect motor windings

Root Causes:
1. Worn motor bearings causing increased mechanical resistance
2. Jammed or misaligned conveyor belt
3. Overloaded hopper — too much tea material causing resistance
4. Electrical fault: phase imbalance or capacitor failure

Immediate Actions:
1. Press EMERGENCY STOP and isolate electrical supply
2. Allow motor to cool for at least 15 minutes before inspection
3. Rotate the drive shaft by hand — resistance indicates mechanical jam
4. Check conveyor belt tension and alignment (acceptable tension: 45–55 N)
5. Inspect motor brushes and commutator for wear

Repair Procedure:
1. Replace motor bearings if noise or stiffness detected (bearing code: 6206-2RS)
2. Re-align conveyor using alignment gauge — max allowed offset: 0.5 mm
3. Clean and clear hopper feed mechanism
4. After repair, run motor at 50% speed for 5 minutes before full production

Preventive Maintenance:
- Lubricate drive shaft bearings every 500 operating hours
- Inspect belt tension monthly
- Motor current baseline check at start of every shift
        """.strip(),
    },
    {
        "doc_id": "builtin/machine_fault_sealing",
        "title": "Sealing Unit Failure — Temperature Fault Diagnosis",
        "fault_type": "machine_fault",
        "sensor": "temperature",
        "content": """
Sealing Heater Fault — Tea Bag Packing Machine

Symptoms:
- Temperature drops below 165°C or spikes above 200°C
- Seal quality degrades — bags not sealing properly or burning
- Temperature sensor reads erratic values

Root Causes:
1. Faulty heating element (open circuit or partial failure)
2. Failed thermocouple (temperature sensor)
3. PID controller fault — not maintaining setpoint
4. Loose electrical connections at heater terminals

Immediate Actions:
1. Stop production if temperature is outside 160–200°C
2. Check PID controller setpoint (should be 180°C for standard tea bags)
3. Verify thermocouple reading against infrared thermometer
4. Inspect heater element resistance (nominal: 22–26 Ω when cold)

Repair Procedure:
1. Replace heating element if resistance is out of range (part: HE-180-TBP)
2. Replace thermocouple if offset > 10°C from reference probe
3. Re-calibrate PID: Kp=1.2, Ki=0.05, Kd=0.15 (standard tea bag settings)
4. Tighten all heater terminal connections to 2.5 Nm torque

Preventive Maintenance:
- Clean sealing jaws weekly (tea dust buildup reduces thermal contact)
- Verify PID setpoint matches product specification before each batch
- Annual calibration of all temperature sensors
        """.strip(),
    },
    {
        "doc_id": "builtin/machine_fault_pressure",
        "title": "Pneumatic Pressure Fault — Air System Diagnosis",
        "fault_type": "machine_fault",
        "sensor": "pressure",
        "content": """
Pneumatic Pressure Fault — Tea Bag Packing Machine

Symptoms:
- Pressure drops below 3.5 bar (normal range: 4.0–5.0 bar)
- Pressure spikes above 6.0 bar
- Erratic bag cutting or forming due to inconsistent air force
- Air leaks audible around fittings or cylinders

Root Causes:
1. Air compressor failure or reduced capacity
2. Leaking pneumatic fittings, hoses, or cylinder seals
3. Blocked air filter reducing supply pressure
4. Pressure regulator malfunction

Immediate Actions:
1. Check compressor pressure gauge — should read 7–8 bar supply
2. Inspect all pneumatic hoses for visible leaks (use soapy water test)
3. Check air filter service indicator (red = replace filter)
4. Verify pressure regulator is set to 4.5 bar (standard setting)

Repair Procedure:
1. Replace leaking hose fittings using push-in connectors (6mm or 8mm)
2. Replace air filter element: part AFF-04-MF (change every 2000 hours)
3. Service pneumatic cylinders: replace O-rings if seals are leaking
4. Replace pressure regulator if unable to hold setpoint: part PR-45-TPM

Preventive Maintenance:
- Drain air receiver tank daily
- Replace air filter every 2000 operating hours or annually
- Lubricate pneumatic cylinders every 1000 hours (food-grade lubricant)
        """.strip(),
    },
    # ── Sensor Fault Docs ────────────────────────────────────────────────
    {
        "doc_id": "builtin/sensor_drift",
        "title": "Sensor Drift — Detection & Calibration Procedure",
        "fault_type": "sensor_drift",
        "sensor": None,
        "content": """
Sensor Drift — Tea Bag Packing Machine

What is Sensor Drift:
Sensor drift occurs when a sensor's output gradually shifts away from the true
physical value over time. Unlike sudden faults, drift is slow and can go unnoticed
until it causes product quality issues.

Signs of Sensor Drift:
- Temperature sensor reading 5–15°C higher/lower than calibration reference
- Pressure sensor consistently offset by 0.3–0.5 bar
- Speed sensor reading does not match visual production count
- Motor current slowly creeping up without any physical change

Detection:
- Compare sensor reading to a calibrated reference instrument
- Track sensor baselines on control charts — gradual upward/downward trend
- AI system flags drift when reading deviates from expected statistical range

Calibration Procedure (General):
1. Stop production and stabilise machine at operating temperature
2. Connect calibrated reference instrument at same measurement point
3. Record sensor reading vs. reference reading (take 5 readings, average)
4. Adjust sensor zero and span using potentiometer or transmitter software
5. Verify calibrated accuracy is within ±0.5% of full scale
6. Document calibration in maintenance log with date and technician name

Re-calibration Trigger Points:
- After sensor replacement
- After any mechanical impact or vibration event
- Every 6 months as scheduled maintenance
- When AI drift alert is triggered
        """.strip(),
    },
    {
        "doc_id": "builtin/sensor_freeze",
        "title": "Sensor Freeze — Diagnosis & Replacement",
        "fault_type": "sensor_freeze",
        "sensor": None,
        "content": """
Sensor Freeze / Stuck Reading — Tea Bag Packing Machine

What is a Sensor Freeze:
A frozen sensor outputs a constant value regardless of actual physical conditions.
This is dangerous because the control system believes conditions are normal
while the true physical state may be changing.

Causes:
1. Failed sensor electronics (ADC or signal conditioning circuit failure)
2. Broken signal cable — open circuit causes last-valid-output to be held
3. PLC input card failure — card stops sampling the sensor channel
4. Software bug in SCADA or PLC that caches last reading

Detection:
- AI system flags freeze when sensor standard deviation = 0 over a sliding window
- Sensor output does not change even when machine conditions change
- Check sensor output during machine start-up — reading should change

Diagnosis Steps:
1. Check sensor cable continuity with multimeter (expected: <2 Ω)
2. Measure sensor output voltage/current at terminal strip:
   - 4–20 mA loop: verify current changes with physical stimulus
   - 0–10V analog: verify voltage changes
3. Swap sensor with known-good spare and observe if reading resumes changing
4. Inspect PLC input card for indicator LEDs (solid red = fault)

Replacement Procedure:
1. Isolate and tag-out machine power
2. Disconnect sensor cable (note wire colors before disconnection)
3. Remove sensor from mounting (typically M12 thread or compression fitting)
4. Install new sensor — apply thread sealant for process fittings
5. Reconnect cable, restore power, verify reading in SCADA
6. Perform calibration check after replacement

Preventive Checks:
- Weekly: visually inspect sensor cables for kinking, chafing, or heat damage
- Monthly: verify all sensor readings change during controlled test
        """.strip(),
    },
    # ── General Maintenance ──────────────────────────────────────────────
    {
        "doc_id": "builtin/general_maintenance",
        "title": "Scheduled Preventive Maintenance Plan — Tea Bag Packing Machine",
        "fault_type": None,
        "sensor": None,
        "content": """
Preventive Maintenance Schedule — Tea Bag Packing Machine

Daily (Every Shift):
- Visual inspection: no leaks, loose guards, or abnormal noise
- Check pneumatic pressure at main gauge (4.0–5.0 bar)
- Verify sealing temperature at HMI matches setpoint (180°C)
- Clean sealing jaws and forming plates (tea dust accumulation)
- Drain air receiver condensate
- Log all sensor baseline readings at start of shift

Weekly:
- Lubricate all lubrication points (grease nipples on conveyor)
- Check belt condition: tension, wear, and alignment
- Inspect all pneumatic hoses for chafing or cracking
- Clean entire machine with compressed air
- Check emergency stop and safety interlock function

Monthly:
- Full bearing inspection (listen for noise, check temperature)
- Verify sensor calibration against reference instruments
- Inspect motor windings with insulation tester (>100 MΩ)
- Check all electrical terminal torques (tighten to specification)
- Verify PID setpoint against product quality results

Annual:
- Full motor service: bearings, brushes, windings test
- Pneumatic cylinder seal replacement (full overhaul kit)
- Temperature sensor calibration certification
- Replace air filter element
- Review and update maintenance log for regulatory compliance

Spare Parts Inventory (Recommended Stock):
- HE-180-TBP: Heating element (min 2 units)
- 6206-2RS: Motor bearing (min 4 units)
- AFF-04-MF: Air filter element (min 2 units)
- Thermocouple type K, M8 thread (min 2 units)
- Conveyor belt section (1 spare length)
        """.strip(),
    },
    {
        "doc_id": "builtin/idle_state",
        "title": "Machine Idle State — Shutdown & Startup Procedures",
        "fault_type": "idle",
        "sensor": None,
        "content": """
Machine Idle / Shutdown State — Tea Bag Packing Machine

Normal Idle Patterns:
- Motor current: 0 A (motor off) or <0.5 A (standby)
- Vibration: <0.2 mm/s
- Speed: 0 bpm
- Temperature: Cooling from operating temp (180°C) toward ambient
- Pressure: May drop to 0 bar if compressor shut down

Planned Shutdown Procedure:
1. Complete current production batch
2. Set speed to 0 at HMI
3. Disable heater — temperature will cool naturally
4. Close air supply valve at main isolator
5. Press MAIN STOP button
6. Isolate electrical supply and apply LOTO (Lock-Out Tag-Out) if maintenance

Startup Procedure (Cold Start):
1. Verify LOTO cleared and all guards in place
2. Restore electrical and pneumatic supply
3. Enable heater — wait for temperature to stabilise at 180°C (≈ 5 minutes)
4. Verify air pressure: 4.0–5.0 bar before starting
5. Run machine at 50% speed for 2 minutes (warm-up cycle)
6. Increase to full production speed (160 bpm)
7. Verify sensor readings are within normal ranges before full production

AI System Behaviour During Idle:
- Detection model flags idle state as anomalous by design
- Operators should acknowledge idle alerts during planned shutdowns
- Consecutive anomaly threshold is disabled during scheduled idle periods
        """.strip(),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# DocumentParser
# ─────────────────────────────────────────────────────────────────────────────
class DocumentParser:
    """
    Parses machine documents into overlapping text chunks ready for embedding.

    Args:
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ───────────────────────────────────────────────────────────
    def parse_builtin_knowledge(self) -> List[Dict[str, Any]]:
        """
        Parse and chunk all built-in maintenance knowledge documents.

        Returns:
            List of chunk dicts ready for embedding:
            {doc_id, chunk_index, title, content, fault_type, sensor}
        """
        all_chunks = []
        for doc in BUILTIN_KNOWLEDGE:
            chunks = self._chunk_text(
                text=doc["content"],
                doc_id=doc["doc_id"],
                title=doc["title"],
                fault_type=doc.get("fault_type"),
                sensor=doc.get("sensor"),
            )
            all_chunks.extend(chunks)
        print(f"[DocumentParser] Parsed {len(BUILTIN_KNOWLEDGE)} documents → "
              f"{len(all_chunks)} chunks.")
        return all_chunks

    def parse_file(
        self,
        file_path: str,
        fault_type: Optional[str] = None,
        sensor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse a text or markdown file into chunks.

        Args:
            file_path:  Absolute or relative path to the .txt or .md file.
            fault_type: Optional fault type tag.
            sensor:     Optional sensor name tag.

        Returns:
            List of chunk dicts.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_id = os.path.basename(file_path)
        title = self._extract_title(text, doc_id)
        return self._chunk_text(text, doc_id=doc_id, title=title,
                                fault_type=fault_type, sensor=sensor)

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        title: str,
        fault_type: Optional[str],
        sensor: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        text = self._clean(text)
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "title": title,
                    "content": chunk_text,
                    "fault_type": fault_type,
                    "sensor": sensor,
                })
                idx += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace and remove markdown artifacts."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_title(text: str, fallback: str) -> str:
        """Try to extract the first heading from text as the document title."""
        match = re.search(r"^#+ (.+)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        first_line = text.strip().splitlines()[0][:100]
        return first_line if first_line else fallback
