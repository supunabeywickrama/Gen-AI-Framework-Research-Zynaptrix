# 📄 Professional PDF Export Feature

> **Export diagnostic conversations as professional maintenance reports with Zynaptrix branding.**

---

## 🎯 Overview

The PDF Export feature allows operators to generate professional diagnostic reports from Central Assistant chat sessions. These reports are structured maintenance documents — not raw chat transcripts — designed for:

- Documentation and compliance
- Sharing with maintenance teams
- Record keeping and audit trails
- Training and knowledge transfer

---

## ✨ Key Features

### 📋 Structured Report Format

Reports are automatically structured into professional sections:

| Section | Description |
|---------|-------------|
| **Header** | Report ID, Machine ID, Date, Status badge |
| **Problem Description** | AI-extracted summary of the reported issue |
| **Diagnosis** | Root cause analysis from the conversation |
| **Solution / Repair Procedure** | Numbered step-by-step instructions |
| **Reference Diagrams** | Images from manual pages (when available) |
| **Operator Notes** | Additional notes (optional) |

### 🏢 Professional Branding

- **Zynaptrix header** with company branding
- **Diagonal watermark** on all pages
- **Confidential footer** with page numbers
- **Status badge** indicating report completion

### 🤖 AI-Powered Content Extraction

The backend uses GPT-4o-mini to intelligently extract:
- Problem summary from user messages
- Diagnosis from assistant responses
- Step-by-step procedures from repair instructions
- Reference images from manual context

---

## 🚀 How to Use

### From Central Assistant

1. Have a diagnostic conversation with the Central Assistant
2. Click the **"Generate Report"** button in the chat header
3. Wait for the progress modal to complete
4. PDF downloads automatically

### Export Button Location

```
┌─────────────────────────────────────────────────────────────┐
│  Central Assistant                    [Generate Report] [X] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Chat conversation...                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Technical Architecture

### Frontend Components

```
frontend/src/
├── professionalReportService.ts   # PDF generation with jsPDF
├── components/
│   └── ExportProgressModal.tsx    # Loading/progress UI
└── store/slices/
    └── copilotSlice.ts            # Export state & thunk
```

### Backend Endpoint

```
GET /api/assistant/sessions/{session_id}/report
```

**Response:**
```json
{
  "sessionId": 123,
  "machineId": "PUMP-001",
  "problemDescription": "The pump is not functioning...",
  "diagnosis": "Root cause analysis indicates...",
  "solutionSteps": [
    "Step 1: Disconnect power",
    "Step 2: Inspect components",
    "..."
  ],
  "images": [
    { "url": "/images/diagram.png", "caption": "Reference Diagram 1" }
  ],
  "timestamp": "2026-04-07T12:00:00",
  "operatorNotes": null
}
```

### PDF Generation Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Button    │ ──▶ │   Backend   │ ──▶ │  GPT-4o-mini │
│   Click     │     │   /report   │     │  Extraction  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    PDF      │ ◀── │   jsPDF     │ ◀── │  Structured  │
│  Download   │     │  Generate   │     │    JSON      │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## 📐 PDF Layout Specifications

### Page Setup
- **Format:** A4 (210mm x 297mm)
- **Margins:** 20mm all sides
- **Orientation:** Portrait

### Color Scheme
| Element | Color | Hex |
|---------|-------|-----|
| Primary (Headers) | Blue | `#1976D2` |
| Header Background | Dark Blue | `#0D47A1` |
| Accent (Steps) | Green | `#10B981` |
| Text | Dark Gray | `#1F2937` |
| Light Background | Light Gray | `#F3F4F6` |

### Typography
- **Title:** Helvetica Bold, 26pt
- **Section Headers:** Helvetica Bold, 11pt
- **Body Text:** Helvetica Normal, 10pt
- **Captions:** Helvetica Italic, 8pt

---

## 🖼️ Image Handling

### Supported Sources
- Manual page images from RAG retrieval
- Base64-encoded images in chat messages
- URL-referenced diagrams

### Image Processing
1. Images are fetched and converted to base64
2. Embedded directly in PDF (no external dependencies)
3. Scaled to fit content width with aspect ratio preserved
4. Bordered with rounded corners
5. Centered captions below each image

---

## 🔧 Dependencies

### Frontend (package.json)
```json
{
  "jspdf": "^2.5.2",
  "jspdf-autotable": "^3.8.4",
  "html2canvas": "^1.4.1"
}
```

### Backend
- OpenAI API (GPT-4o-mini for content extraction)
- Existing FastAPI infrastructure

---

## 📁 Output Files

### Filename Format
```
Diagnostic_Report_{SESSION_ID}_{DATE}.pdf
```

**Example:** `Diagnostic_Report_123_2026-04-07.pdf`

### File Location
Downloads to user's default download directory.

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Garbled text | Unicode characters in jsPDF | Use ASCII-only text (fixed) |
| Empty sections | AI extraction failed | Check backend logs |
| No images | Images not in chat | Ensure manual context is used |
| Export fails | Network error | Check backend is running |

### Debug Logging

Backend logs extraction progress:
```
INFO: Report generation - Session 123: 5 messages, 2 images
INFO: AI Response for report: {"problem": "...", ...}
INFO: Report generated - Problem: The pump is not...
```

---

## 📝 Future Enhancements

- [ ] Custom report templates
- [ ] Multi-language support
- [ ] Email export option
- [ ] Batch export for multiple sessions
- [ ] Digital signature support
- [ ] QR code linking to session

---

*Feature developed for Zynaptrix Industrial Copilot Platform — April 2026*
