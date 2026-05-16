# Estrategias de Extracción de Metadata de PDFs

Documentación de las estrategias utilizadas para extraer **título** y **abstract** de PDFs de arXiv usando dos librerías: **PyMuPDF** y **Docling**.

## 1. Overview

### Objetivo
Extraer dos campos clave de cada PDF:
- **Título**: Nombre del paper (primeras líneas antes del abstract)
- **Abstract**: Resumen técnico del trabajo (sección identificada)

### Entrada
- PDFs en `data/pdfs/{categoria}/{arxiv_id}.pdf`
- Metadata en `data/raw/metadata.json` con campos: `arxiv_id`, `title` (como fallback), `primary_category`

### Salida
- **PyMuPDF**: `data/interim/extracted_pymupdf.json`
- **Docling**: `data/interim/extracted_docling.json`

Esquema común:
```json
{
  "arxiv_id": "2301.00001",
  "category": "cs.AI",
  "title": "Deep Learning for Vision",
  "abstract_pymupdf": "We propose...",
  "extraction_method": "pattern_based|fallback|null",
  "status": "success|failed"
}
```

## 2. Extracción del Título

### 2.1 Estrategia General

El título se encuentra en las **primeras líneas del PDF**, antes del abstract.

**Algoritmo:**
1. Dividir texto en líneas
2. Recolectar líneas hasta encontrar:
   - Una línea vacía, O
   - La palabra clave "Abstract"
3. Limitar a máximo 5 líneas
4. Usar las primeras 2 líneas como título probable
5. Aplicar limpieza de texto

**Código base (ambas librerías):**

```python
def extract_title_from_text(text: str) -> str:
    """Extrae el título del PDF (primeras líneas antes de Abstract)."""
    lines = text.split("\n")
    title_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:                           # Línea vacía
            continue
        if stripped.lower().startswith("abstract"): # Encontró Abstract
            break
        title_lines.append(stripped)
        if len(title_lines) >= 5:                  # Máximo 5 líneas
            break
    
    # Tomar primeras 2 líneas como título
    if title_lines:
        return " ".join(title_lines[:2]).strip()
    return ""
```

### 2.2 Ejemplo Real

**Texto bruto del PDF:**

```
Deep Learning for Vision Tasks
A Comprehensive Survey

Abstract
Deep learning has revolutionized computer vision...
```

**Extracción:**

| Paso | Resultado |
|------|-----------|
| Línea 1 | `"Deep Learning for Vision Tasks"` |
| Línea 2 | `"A Comprehensive Survey"` |
| Línea 3 | (vacía, termina recolección) |
| **Título final** | `"Deep Learning for Vision Tasks A Comprehensive Survey"` |

### 2.3 Limitaciones

| Caso Problemático | Resultado | Mitigación |
|-------------------|-----------|-----------|
| Título multilínea largo | Captura solo primeras 2 líneas | Suficiente para identificación |
| Sin encabezado clara (todo párrafo) | Captura demasiado texto | Fallback a metadata.json |
| PDF escaneado sin OCR | Nada extraído | `title` = valor de metadata |

### 2.4 Fallback

Si no se extrae título del PDF:
```python
if not result["title"]:
    result["title"] = article.get("title", "")  # Usar metadata
```

## 3. Extracción del Abstract

### 3.1 Estrategia Principal: Búsqueda por Patrón

**Objetivo**: Encontrar la sección "Abstract" e identificar dónde termina.

#### 3.1.1 Delimitación del INICIO

El abstract comienza después de la palabra clave "Abstract" como encabezado.

**Patrón Regex:**

```regex
(?:^|\n)\s*abstract\s*(?:[:—\-])?\s*\n
```

**Desglose:**

| Componente | Significa | Ejemplo |
|-----------|-----------|---------|
| `(?:^|\n)` | Inicio de línea o string | Después de `\n` |
| `\s*` | Espacios/tabs opcionales | `  abstract` |
| `abstract` | Palabra literal (case-insensitive) | "Abstract", "ABSTRACT" |
| `(?:[:—\-])?` | Separador opcional | `Abstract:`, `Abstract —` |
| `\s*\n` | Espacios + salto de línea | Fin del encabezado |

**Ejemplos que coinciden:**

```
✓ Abstract
✓ ABSTRACT
✓ Abstract:
✓ Abstract —
✓ Abstract -
✓   Abstract  
```

#### 3.1.2 Delimitación del FIN

El abstract termina cuando encuentra una de estas palabras clave (siguiente sección):

```python
introduction|keywords|references|[0-9]+\.|conclusion|related|acknowledgment
```

**Por qué estas palabras?**

| Palabra Clave | Ubicación Típica | Razón |
|--------------|-----------------|-------|
| `introduction` | Primera sección formal | Casi siempre presente |
| `keywords` | Después del abstract | Muchos papers |
| `references` | Final del paper | Si no hay intro |
| `[0-9]+\.` | Numeración de secciones | `1. Introduction` |
| `conclusion` | Para papers cortos | Si falta intro |
| `related` | "Related Work" | Variante de intro |
| `acknowledgment` | Antes de referencias | Algunos papers |

#### 3.1.3 Patrón Completo

```python
match = re.search(
    r'(?:^|\n)\s*abstract\s*(?:[:—\-])?\s*\n(.*?)(?=\n\s*(?:introduction|keywords|references|[0-9]+\.|conclusion|related|acknowledgment)|\Z)',
    text,
    re.DOTALL | re.IGNORECASE
)

if match:
    abstract = match.group(1).strip()  # Captura el grupo 1: contenido
    return abstract, "pattern_based"
```

**Flags de regex:**
- `re.DOTALL`: `.` también coincide con `\n` (multilinea)
- `re.IGNORECASE`: "abstract" = "Abstract" = "ABSTRACT"

#### 3.1.4 Ejemplo Paso a Paso

**Texto bruto:**

```
Abstract
Deep learning models have achieved remarkable results in computer vision.
We propose a novel architecture that...

Introduction
In this work, we address the problem of...
```

**Ejecución del regex:**

1. **Encuentra**: `(?:^|\n)\s*abstract` → línea que empieza con "Abstract"
2. **Inicio de captura**: Después de `\s*\n` (salto de línea post-"Abstract")
3. **Captura contenido**: `(.*?)` → "Deep learning models have achieved...\nWe propose..."
4. **Fin de captura**: `(?=\n\s*introduction)` → lookahead encuentra "Introduction"
5. **Resultado**: 
```
"Deep learning models have achieved remarkable results in computer vision.
We propose a novel architecture that..."
```

### 3.2 Estrategia de Fallback

Si el patrón **no coincide** (abstract mal estructurado):

```python
else:
    # Fallback: primeros 1500 caracteres
    abstract = text[:1500].strip()
    return abstract, "fallback"
```

**Casos donde se activa el fallback:**

| Caso | Razón |
|------|-------|
| Abstract sin encabezado claro | "Abstract" no está como línea separada |
| Formato atípico | Construcción rara del PDF |
| PDF escaneado sin OCR | Sin texto extraíble |
| Extracto de página escanneada | No hay estructura de texto |

**Ventaja**: Siempre extrae algo (1500 primeros caracteres)  
**Desventaja**: Puede incluir más que el abstract (ej: encabezado, autores)

### 3.3 Limpieza del Texto

Después de extraer (patrón o fallback), se aplica normalización:

```python
def clean_text(text: str) -> str:
    # 1. Unir líneas partidas por guión (hyphenation)
    text = re.sub(r"-\n", "", text)
    
    # 2. Reemplazar saltos de línea con espacios
    text = re.sub(r"\n", " ", text)
    
    # 3. Normalizar espacios múltiples
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()
```

**Paso a paso:**

```
Entrada (raw):
"Deep learning models have achieved re-
markable results on multiple tasks. We propose
a  novel  architecture  for..."

Paso 1 (guiones):
"Deep learning models have achieved remarkable results on multiple tasks. We propose
a  novel  architecture  for..."

Paso 2 (saltos de línea):
"Deep learning models have achieved remarkable results on multiple tasks. We propose a  novel  architecture  for..."

Paso 3 (espacios múltiples):
"Deep learning models have achieved remarkable results on multiple tasks. We propose a novel architecture for..."

Salida: "Deep learning models have achieved remarkable results on multiple tasks. We propose a novel architecture for..."
```

**Transformaciones:**

| Patrón | Busca | Reemplaza | Ejemplo |
|--------|-------|-----------|---------|
| `-\n` | Guión + salto | Nada | `"deep-\nlearning"` → `"deeplearning"` |
| `\n` | Salto de línea | Espacio | `"line1\nline2"` → `"line1 line2"` |
| `\s+` | 1+ espacios | Un espacio | `"word1  word2"` → `"word1 word2"` |

## 4. Implementación por Librería

### 4.1 PyMuPDF (fitz)

**Ventajas:**
- Rápido
- No requiere modelos adicionales
- Extrae texto nativo del PDF

**Desventajas:**
- No maneja bien PDFs escaneados (sin OCR)
- Puede fragmentar texto

#### 4.2.1 Extracción de Título (Markdown + Validación)

```python
def extract_title_from_markdown(markdown_text: str) -> str:
    """
    1. Busca primer encabezado (#)
    2. Valida que NO sea palabra clave (Abstract, Introduction, etc.)
    3. Fallback a texto plano si falla
    """
    lines = markdown_text.split("\n")
    
    for line in lines:
        if line.startswith("#"):
            # Extraer texto después de los #
            title = re.sub(r"^#+\s*", "", line).strip()
            
            # Validar: NO debe ser sección clave
            section_keywords = r"\b(abstract|introduction|keywords|...)\b"
            if not re.search(section_keywords, title, re.IGNORECASE):
                return title  # Título válido
    
    # Fallback: usar estrategia PyMuPDF en texto plano
    return extract_title_from_text(markdown_text)
```

**Ejemplo:**

```
Markdown del PDF:
# Deep Learning for Vision Tasks
# A Comprehensive Survey

## Abstract
Deep learning has revolutionized...

Extracción:
1. Encuentra primer #: "Deep Learning for Vision Tasks"
2. Valida: NO es "abstract", "introduction", etc.
3. Retorna: "Deep Learning for Vision Tasks"
```

**Validación regex:**
- Rechaza: "Abstract", "ABSTRACT", "Introduction", "Keywords"
- Acepta: "Deep Learning for Vision Tasks", "A Survey on..."

#### 4.2.2 Extracción de Abstract (Markdown + Validación)

```python
def extract_abstract_from_markdown(markdown_text: str) -> tuple[str, str | None]:
    """
    Estrategia combinada para máxima confiabilidad:
    
    1. Busca encabezados (#, ##, ###) que contengan "abstract"
    2. Valida con regex que sea realmente "abstract" (no "abstract_method")
    3. Extrae contenido hasta siguiente encabezado
    4. Valida longitud mínima (> 50 caracteres)
    5. Fallback a pattern matching si falla
    """
    lines = markdown_text.split("\n")
    
    abstract_start = None
    abstract_end = None
    
    for i, line in enumerate(lines):
        if line.startswith("#"):
            # Extraer texto después de los #
            heading_text = re.sub(r"^#+\s*", "", line)
            
            # Validar: palabra completa "abstract"
            if re.search(r"\babstract\b", heading_text, re.IGNORECASE):
                abstract_start = i + 1
                
                # Buscar fin: siguiente encabezado
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("#"):
                        abstract_end = j
                        break
                
                if abstract_end is None:
                    abstract_end = len(lines)
                break
    
    # Si encontró con markdown structure
    if abstract_start is not None:
        abstract_text = "\n".join(lines[abstract_start:abstract_end]).strip()
        
        # Validar longitud mínima (evitar falsos positivos)
        if abstract_text and len(abstract_text) > 50:
            return abstract_text, "markdown_structure"
    
    # Fallback: no funcionó markdown, usar pattern matching (PyMuPDF logic)
    plain_text = re.sub(r"^#+\s+", "", markdown_text, flags=re.MULTILINE)
    return extract_abstract_from_text(plain_text)
```

**Ejemplo:**

```
Markdown del PDF:
# Deep Learning for Vision Tasks
# A Comprehensive Survey

## Abstract
Deep learning models have achieved remarkable results in computer vision.
We propose a novel architecture that...

## Introduction
In this work, we address the problem of...

Extracción:
1. Itera líneas buscando #
2. Encuentra "## Abstract"
3. Valida: contiene palabra "abstract" ✓
4. abstract_start = siguiente línea después de "## Abstract"
5. Busca siguiente #: "## Introduction"
6. abstract_end = línea de "## Introduction"
7. Retorna contenido entre abstract_start y abstract_end
8. extraction_method = "markdown_structure"
```

**Validación regex:**
- `\babstract\b`: Palabra completa (rechaza "abstract_method", "abstract_concepts")
- `re.IGNORECASE`: Acepta "Abstract", "ABSTRACT", "AbStRaCt"
- Longitud mínima 50 caracteres: Evita capturar títulos cortos como abstract

---

### 4.2.7 Comparación de Estrategias

#### **PyMuPDF (Texto Plano)**

**Método:**
- Exporta texto bruto (`get_text()`)
- Busca "Abstract" con regex simple
- Extrae hasta siguiente sección

**Ventajas:**
- ⚡ Muy rápido
- 📦 Sin modelos adicionales
- 🔄 Fallback simple

**Desventajas:**
- ❌ No valida estructura
- ❌ Sensible a formato del PDF
- ❌ Falla con PDFs mal estructurados
- ❌ Puede capturar texto erróneamente

---

#### **Docling (Markdown + Validación Regex)**

**Método:**
- Exporta markdown (`export_to_markdown()`)
- Busca encabezados (`#`)
- Valida con regex palabra completa
- Valida longitud mínima
- Fallback a pattern matching

**Ventajas:**
- ✅ Preserva estructura del documento
- ✅ Valida a múltiples niveles
- ✅ Rechaza falsos positivos
- ✅ Fallback integrado
- ✅ **Robusto con:**
  - PDFs con encabezados numerados (1. Abstract)
  - Múltiples "Abstract" en documento
  - Texto erróneamente convertido a `#`

**Desventajas:**
- 🐢 Más lento que PyMuPDF
- 💾 Requiere más memoria
- 📥 Descarga modelos (primera vez)

---

#### **Tabla Comparativa**

| Aspecto | PyMuPDF | Docling |
|--------|---------|---------|
| **Velocidad** | ⚡⚡⚡ Muy rápido | ⚡⚡ Medio |
| **Estructura** | ❌ Texto plano | ✅ Markdown |
| **Validaciones** | 1 (regex básico) | 3 (estructura + regex + longitud) |
| **Fallback** | Simple (1500 chars) | Inteligente (usa PyMuPDF logic) |
| **OCR** | ❌ No | ✅ Sí (automático o forzado) |
| **GPU** | ❌ No | ✅ Sí (para OCR) |
| **Robusto** | Media | Alta |
| **Memoria** | 💾 Bajo | 💾💾 Alto |
| **Modelos** | ❌ No | ✅ Auto-descarga |

---

#### **Casos de Uso**

**Usa PyMuPDF si:**
- PDFs de arXiv típicos (bien estructurados)
- Velocidad es crítica
- Memoria limitada
- Recursos computacionales escasos

**Usa Docling si:**
- PDFs con layout complejo
- Algunos PDFs escaneados
- Tienes GPU disponible
- Necesitas máxima precisión
- PDFs "sucios" o mal estructurados

---

### 4.4 Código de extracción PyMuPDF

```python
import fitz

def process_pdf_pymupdf(pdf_path: Path) -> dict:
    try:
        doc = fitz.open(pdf_path)
        
        # Extraer texto de primeras 2 páginas
        text = ""
        for page_num in range(min(2, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
        
        if not text.strip():
            return {"status": "failed", "abstract": "", "title": ""}
        
        # Extraer y limpiar
        abstract, method = extract_abstract_from_text(text)
        abstract = clean_text(abstract)
        title = extract_title_from_text(text)
        title = clean_text(title)
        
        return {
            "title": title,
            "abstract_pymupdf": abstract,
            "extraction_method": method,
            "status": "success"
        }
    except Exception as e:
        return {"status": "failed", "abstract": "", "title": ""}
```

**Uso:**

```bash
uv run python src/extraction/extract_pymupdf.py
```

### 4.2 Docling

**Ventajas:**
- Detección automática de layout
- Exporta a markdown (estructura preservada)
- Soporte para OCR en PDFs escaneados
- Mejor manejo de estructuras complejas
- GPU acceleration para OCR

**Desventajas:**
- Más lento que PyMuPDF
- Requiere descargar modelos (primera ejecución)
- Requiere más memoria
- A veces convierte texto erróneo a encabezados markdown

---

#### 4.2.1 Estrategia: Markdown Structure + Validación Regex

**Diferencia clave con PyMuPDF:**

Docling exporta a **markdown preservando la estructura del documento** (`export_to_markdown()`). Esto permite una estrategia combinada más confiable basada en encabezados:

| Aspecto | PyMuPDF | Docling |
|--------|---------|---------|
| **Extracción** | Texto plano | Markdown con `#` |
| **Búsqueda** | Regex simple en texto | Búsqueda de encabezados |
| **Validación** | Mínima | Regex + longitud + estructura |
| **Robustez** | Media | Alta |

**Pipeline de Docling:**

```
PDF → Docling converter → Markdown (preserva estructura)
                              ↓
                    1. Busca encabezados (#)
                    2. Valida con regex
                    3. Valida longitud
                              ↓
                         Extrae abstract
                    (si todo falla: fallback)
```

---

#### 4.2.2 Extracción de Título (Markdown + Validación)

```python
def extract_title_from_markdown(markdown_text: str) -> str:
    """
    Estrategia combinada para extraer título del markdown:
    
    1. Busca primer encabezado (#, ##, ###, etc.)
    2. Valida que NO sea palabra clave (abstract, introduction, etc.)
    3. Retorna el título válido
    4. Fallback a estrategia PyMuPDF si falla
    """
    lines = markdown_text.split("\n")
    
    # Palabras clave que indican secciones, no títulos
    section_keywords = r"\b(abstract|introduction|keywords|references|conclusion|related|acknowledgment)\b"
    
    for line in lines:
        # Detectar encabezado: línea que empieza con #
        if line.startswith("#"):
            # Extraer texto después de los #
            # Ejemplo: "## Abstract" → "Abstract"
            title = re.sub(r"^#+\s*", "", line).strip()
            
            # Validar con regex: NO debe ser palabra clave
            if not re.search(section_keywords, title, re.IGNORECASE):
                return title  # Título válido encontrado
    
    # Fallback: no encontró con markdown, usar estrategia PyMuPDF
    return extract_title_from_text(markdown_text)
```

**Ejemplo paso a paso:**

```
Markdown:
# Deep Learning for Vision Tasks
# A Comprehensive Survey

## Abstract
Deep learning has revolutionized...

Ejecución:
1. Lee línea: "# Deep Learning for Vision Tasks"
2. Extrae: "Deep Learning for Vision Tasks"
3. Valida: NO coincide con "abstract", "introduction", etc. ✓
4. Retorna: "Deep Learning for Vision Tasks"
5. extraction_method: (implícito en título)
```

**Casos de validación:**

```python
# Acepta (no son palabras clave):
"# Deep Learning for Vision"          → ✓ Válido
"# A Novel Approach to NLP"           → ✓ Válido
"## Novel Abstract Framework"         → ✓ Válido (abstract es adjetivo)

# Rechaza (son palabras clave):
"# Abstract"                          → ❌ Es sección
"## Introduction to Deep Learning"    → ❌ Comienza con keyword
"### Keywords and Concepts"           → ❌ Comienza con keyword
```

---

#### 4.2.3 Extracción de Abstract (Markdown + Validación Regex)

```python
def extract_abstract_from_markdown(markdown_text: str) -> tuple[str, str | None]:
    """
    Estrategia combinada para máxima confiabilidad:
    
    1. Busca encabezados (#, ##, ###) que contengan palabra "abstract"
    2. Valida con regex palabra completa (rechaza "abstract_method")
    3. Extrae contenido hasta siguiente encabezado
    4. Valida longitud mínima (> 50 caracteres)
    5. Fallback a pattern matching si falla
    
    Returns:
        (abstract_text, extraction_method)
        extraction_method: "markdown_structure" | "pattern_based" | "fallback" | None
    """
    lines = markdown_text.split("\n")
    
    abstract_start = None
    abstract_end = None
    
    # Buscar encabezado que sea "abstract"
    for i, line in enumerate(lines):
        if line.startswith("#"):
            # Extraer texto después de los #
            # Ejemplo: "## Abstract" → "Abstract"
            heading_text = re.sub(r"^#+\s*", "", line).strip()
            
            # Validar: palabra completa "abstract" (no "abstract_method")
            if re.search(r"\babstract\b", heading_text, re.IGNORECASE):
                abstract_start = i + 1
                
                # Buscar fin: siguiente encabezado (cualquier #)
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("#"):
                        abstract_end = j
                        break
                
                # Si no hay siguiente encabezado, es hasta el final
                if abstract_end is None:
                    abstract_end = len(lines)
                break
    
    # Extraer contenido si encontró sección "Abstract"
    if abstract_start is not None:
        abstract_text = "\n".join(lines[abstract_start:abstract_end]).strip()
        
        # Validar longitud mínima (evitar capturar títulos cortos)
        if abstract_text and len(abstract_text) > 50:
            return abstract_text, "markdown_structure"
    
    # Fallback: markdown structure falló, usar pattern matching (PyMuPDF logic)
    # Convertir markdown a plain text removiendo formatos
    plain_text = re.sub(r"^#+\s+", "", markdown_text, flags=re.MULTILINE)
    plain_text = re.sub(r"\*\*(.+?)\*\*", r"\1", plain_text)  # bold
    plain_text = re.sub(r"\*(.+?)\*", r"\1", plain_text)       # italic
    
    return extract_abstract_from_text(plain_text)
```

**Ejemplo paso a paso:**

```
Markdown:
# Deep Learning for Vision Tasks
# A Comprehensive Survey

## Abstract
Deep learning models have achieved remarkable results in computer vision.
We propose a novel architecture that...
The experimental validation demonstrates superior performance.

## Introduction
In this work, we address the problem of...

Ejecución:
1. Itera líneas buscando #
2. Línea 5: "## Abstract"
3. Extrae: "Abstract"
4. Valida: contiene palabra completa "abstract" ✓
5. abstract_start = 6 (siguiente línea)
6. Busca siguiente #: encontrado en "## Introduction" (línea 9)
7. abstract_end = 9
8. Extrae líneas 6-8:
   "Deep learning models have achieved...\n..."
9. Valida longitud: > 50 caracteres ✓
10. Retorna: texto completo, extraction_method="markdown_structure"
```

**Validaciones aplicadas:**

| Validación | Propósito | Ejemplo |
|-----------|----------|---------|
| `\babstract\b` | Palabra completa | Acepta "Abstract", rechaza "abstract_method" |
| `re.IGNORECASE` | Case-insensitive | "ABSTRACT" = "Abstract" = "abstract" |
| Longitud > 50 | Evitar falsos positivos | Rechaza si es muy corto (no es abstract real) |
| Siguiente `#` | Delimitación de fin | Para cuando encuentra siguiente sección |

---

#### 4.2.4 Código Completo de process_pdf (Docling)

```python
def process_pdf(
    pdf_path: Path,
    converter_no_ocr: DocumentConverter,
    converter_with_ocr: DocumentConverter
) -> dict:
    """
    Procesa un PDF con Docling usando:
    1. Conversión a markdown (preserva estructura)
    2. Intenta sin OCR primero (rápido)
    3. Usa OCR si no extrae suficiente texto
    4. Extrae título y abstract con validaciones
    5. Reporta método de extracción usado
    """
    try:
        # Intenta sin OCR primero
        result = converter_no_ocr.convert(pdf_path)
        markdown_text = result.document.export_to_markdown()  # ← Markdown, no texto plano
        ocr_used = False

        # Si no extrae suficiente, reintentar con OCR
        if not markdown_text.strip() or len(markdown_text.strip()) < 100:
            try:
                result = converter_with_ocr.convert(pdf_path)
                markdown_text = result.document.export_to_markdown()
                ocr_used = True
            except Exception:
                pass

        if not markdown_text.strip():
            return {
                "title": "",
                "abstract_docling": "",
                "extraction_method": None,
                "ocr_used": False,
                "status": "failed",
            }

        # Extracción con estrategia markdown + regex
        abstract, method = extract_abstract_from_markdown(markdown_text)
        abstract = clean_text(abstract)

        title = extract_title_from_markdown(markdown_text)
        title = clean_text(title)

        return {
            "title": title,
            "abstract_docling": abstract,
            "extraction_method": method,  # "markdown_structure" | "pattern_based" | "fallback"
            "ocr_used": ocr_used,
            "status": "success",
        }
    except Exception as e:
        return {
            "title": "",
            "abstract_docling": "",
            "extraction_method": None,
            "ocr_used": False,
            "status": "failed",
        }
```

---

#### 4.2.5 Detección Automática de GPU

```python
def detect_gpu() -> bool:
    """
    Detecta si hay GPU disponible (NVIDIA CUDA).
    
    Intenta:
    1. torch.cuda.is_available() (si PyTorch instalado)
    2. nvidia-smi (fallback manual)
    3. Retorna False si nada disponible
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass

    # Fallback: intentar nvidia-smi
    import subprocess
    try:
        subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=True,
            timeout=5
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, TimeoutError):
        pass

    return False


# Uso en configuración
gpu_available = detect_gpu()

if args.force_ocr:
    # OCR forzado en todos los PDFs
    ocr_options = EasyOcrOptions(use_gpu=gpu_available)
    converter_options = PdfPipelineOptions(ocr_options=ocr_options)
    converter = DocumentConverter(pdf_pipeline_options=converter_options)
else:
    # Detección inteligente: sin OCR primero, OCR si es necesario
    converter_no_ocr = DocumentConverter()
    
    ocr_options = EasyOcrOptions(use_gpu=gpu_available)
    converter_options = PdfPipelineOptions(ocr_options=ocr_options)
    converter_with_ocr = DocumentConverter(pdf_pipeline_options=converter_options)
```

---

#### 4.2.6 Uso de extract_docling.py

```bash
# Detección automática de OCR (sin OCR → con OCR si es necesario)
uv run python src/extraction/extract_docling.py

# Forzar OCR en todos los PDFs (más lento, mejor para escaneos)
uv run python src/extraction/extract_docling.py --force-ocr
```

**Salida esperada:**

```
GPU disponible: Sí
Detección automática de OCR (sin OCR → con OCR si es necesario)

Reanudando: 3 artículos ya procesados

[1/2] 2301.00001 (cs.AI)... OK
[2/2] 2301.00002 (cs.AI)... OK (OCR)

==============================================================
Guardado: data/interim/extracted_docling.json
Nuevos extraídos: 2 | Nuevos fallos: 0
Usó OCR: 1/2
Total acumulado: 5
==============================================================
```

## 5. Reanudación de Procesamiento

Ambos scripts implementan reanudación automática:

```python
# Leer resultados existentes
if output_path.exists():
    existing_results = json.loads(output_path.read_text())
    processed_ids = {r["arxiv_id"] for r in existing_results}
else:
    existing_results = []
    processed_ids = set()

# Filtrar artículos no procesados
metadata_to_process = [a for a in metadata if a["arxiv_id"] not in processed_ids]

# ... procesar solo los nuevos ...

# Combinar y guardar
all_results = existing_results + results
output_path.write_text(json.dumps(all_results, ...))
```

**Comportamiento:**
- Ejecuta varias veces → continúa donde paró
- Si falla a mitad → reintenta desde siguiente artículo
- No reprocesa artículos ya completados

## 6. Comparación de Métodos

### 6.1 Tabla Comparativa

| Aspecto | PyMuPDF | Docling |
|--------|---------|---------|
| **Velocidad** | ⚡⚡⚡ Rápido | ⚡⚡ Medio |
| **Calidad (nativo)** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **OCR capability** | ❌ No | ✅ Sí |
| **GPU support** | ❌ No | ✅ Sí |
| **Memoria** | 💾 Bajo | 💾💾 Alto |
| **Modelos req.** | ❌ No | ✅ Sí (auto) |
| **PDFs escaneados** | ❌ Mal | ✅⚡ Bien |

### 6.2 Casos de Uso

**Usa PyMuPDF si:**
- PDFs de arXiv típicos (texto nativo)
- Necesitas velocidad pura
- Memoria limitada
- No hay PDFs escaneados

**Usa Docling si:**
- PDFs con layout complejo
- Algunos PDFs escaneados (OCR)
- Tienes GPU para OCR
- Quieres máxima precisión

## 7. Campos de Salida y Auditoría

### 7.1 Campos en JSON de Extracción

```json
{
  "arxiv_id": "2301.00001",
  "category": "cs.AI",
  "title": "Deep Learning for Vision Tasks",
  "abstract_pymupdf": "We propose...",
  "abstract_docling": "Deep learning models...",
  "extraction_method": "markdown_structure|pattern_based|fallback|null",
  "ocr_used": false,
  "status": "success|failed"
}
```

| Campo | Valores | Significado |
|-------|---------|------------|
| `arxiv_id` | string | ID de arXiv del artículo |
| `category` | string | Categoría primaria (cs.AI, etc.) |
| `title` | string | Título extraído del PDF |
| `abstract_*` | string | Abstract extraído por cada método |
| `extraction_method` | ver abajo | Método usado para extraer abstract |
| `ocr_used` | bool | Si se usó OCR (Docling) |
| `status` | success\|failed | Si la extracción fue exitosa |

### 7.2 Métodos de Extracción (extraction_method)

**Para PyMuPDF:**
- `"pattern_based"` → Encontró "Abstract" con regex (recomendado)
- `"fallback"` → Usó primeros 1500 caracteres (menos confiable)
- `null` → No se pudo extraer nada

**Para Docling:**
- `"markdown_structure"` → Encontró `# Abstract` en markdown (más confiable)
- `"pattern_based"` → Fallback a regex de PyMuPDF
- `"fallback"` → Primeros 1500 caracteres
- `null` → No se pudo extraer nada

**Auditoría:**
```python
# Verificar distribución de métodos
import json
from collections import Counter

data = json.load(open("data/interim/extracted_docling.json"))
methods = Counter(r.get("extraction_method") for r in data)

print("Distribución de métodos:")
for method, count in methods.most_common():
    print(f"  {method}: {count}")
    
# Resultado esperado para Docling:
#   markdown_structure: 1850  (92%) ← Excelente
#   pattern_based: 130        (6%)  ← Fallback aceptable
#   fallback: 15              (1%)  ← Problemas
#   None: 5                   (0%)  ← Fallos totales
```

---

## 8. Evaluación y Comparación

### 8.1 Comparación con Ground Truth

Después de extracción, comparar con `abstract_api` de metadata.json (ground truth de arXiv):

```python
from difflib import SequenceMatcher

def similarity_ratio(a: str, b: str) -> float:
    """Calcula similitud entre dos strings (0.0 a 1.0)."""
    return SequenceMatcher(None, a, b).ratio()

# Ejemplo de análisis
data = json.load(open("data/interim/extracted_docling.json"))
metadata = json.load(open("data/raw/metadata.json"))

# Crear lookup de metadata
meta_by_id = {m["arxiv_id"]: m for m in metadata}

# Calcular similitud
results = []
for record in data:
    arxiv_id = record["arxiv_id"]
    meta = meta_by_id[arxiv_id]
    
    sim_pymupdf = similarity_ratio(
        record.get("abstract_pymupdf", ""),
        meta.get("abstract_api", "")
    )
    sim_docling = similarity_ratio(
        record.get("abstract_docling", ""),
        meta.get("abstract_api", "")
    )
    
    results.append({
        "arxiv_id": arxiv_id,
        "method_pymupdf": record.get("extraction_method_pymupdf"),
        "similarity_pymupdf": sim_pymupdf,
        "similarity_docling": sim_docling,
        "better_method": "docling" if sim_docling > sim_pymupdf else "pymupdf"
    })

# Estadísticas
import statistics
sims_pymupdf = [r["similarity_pymupdf"] for r in results]
sims_docling = [r["similarity_docling"] for r in results]

print(f"PyMuPDF   - Media: {statistics.mean(sims_pymupdf):.3f}, "
      f"Mediana: {statistics.median(sims_pymupdf):.3f}")
print(f"Docling   - Media: {statistics.mean(sims_docling):.3f}, "
      f"Mediana: {statistics.median(sims_docling):.3f}")

# Contar victorias
wins_docling = sum(1 for r in results if r["better_method"] == "docling")
print(f"\nDocling gana en: {wins_docling}/{len(results)} casos")
```

### 8.2 Métricas de Interés

```python
# Análisis por extraction_method
methods_performance = {}

for method in ["markdown_structure", "pattern_based", "fallback"]:
    records = [r for r in data if r.get("extraction_method") == method]
    if records:
        sims = [
            similarity_ratio(r.get("abstract_docling", ""), 
                           meta_by_id[r["arxiv_id"]].get("abstract_api", ""))
            for r in records
        ]
        methods_performance[method] = {
            "count": len(records),
            "avg_similarity": statistics.mean(sims),
            "min_similarity": min(sims),
            "max_similarity": max(sims)
        }

print("Rendimiento por método:")
for method, metrics in methods_performance.items():
    print(f"\n{method}:")
    print(f"  Cantidad: {metrics['count']}")
    print(f"  Similitud promedio: {metrics['avg_similarity']:.3f}")
    print(f"  Rango: {metrics['min_similarity']:.3f} - {metrics['max_similarity']:.3f}")
```

### 8.3 Decisión Final

Basado en similitud con `abstract_api`:

```python
# Elegir mejor método por artículo
for arxiv_id, meta in meta_by_id.items():
    pymupdf_sim = similarity_ratio(pymupdf_abstracts[arxiv_id], meta["abstract_api"])
    docling_sim = similarity_ratio(docling_abstracts[arxiv_id], meta["abstract_api"])
    
    # Usar el mejor
    final_abstract = (docling_abstracts[arxiv_id] if docling_sim > pymupdf_sim 
                     else pymupdf_abstracts[arxiv_id])
    final_method = "docling" if docling_sim > pymupdf_sim else "pymupdf"
```

## 9. Medición de Tiempos de Procesamiento

Ambos scripts (`extract_pymupdf.py` y `extract_docling.py`) incluyen medición automática de tiempos para análisis de rendimiento.

### 9.1 ¿Qué se Mide?

#### **extract_pymupdf.py**

Se miden **dos fases** separadas por PDF:

```
extraccion_texto_seg: Tiempo que tarda PyMuPDF en leer el PDF y extraer texto
                      (incluye fitz.open() + get_text() para 2 primeras páginas)
                      
procesamiento_seg: Tiempo que tarda la lógica de limpieza y búsqueda del abstract
                  (incluye save_raw_text + extract_abstract + clean_text + extract_title)
                  
total_seg: Suma de extraccion_texto_seg + procesamiento_seg
```

#### **extract_docling.py**

Se miden **dos fases** separadas por PDF:

```
conversion_seg: Tiempo que tarda Docling en convertir el PDF a markdown
               (incluye converter.convert() + export_to_markdown())
               
procesamiento_seg: Tiempo que tarda la lógica de extracción del abstract
                  (incluye save_raw_markdown + extract_abstract + clean_text + extract_title)
                  
total_seg: Suma de conversion_seg + procesamiento_seg
```

### 9.2 Implementación Técnica

**Herramienta usada:** `time.perf_counter()`
- Reloj monotónico de alta resolución
- No se ve afectado por ajustes del reloj del sistema
- Precisión: microsegundos
- Redondeo: 4 decimales

**Ejemplo de medición en extract_pymupdf.py:**

```python
def process_pdf(pdf_path: Path, raw_dir: Path) -> dict:
    try:
        # Medir extracción de texto
        tiempo_inicio_extraccion = time.perf_counter()
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(min(2, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
        tiempo_extraccion_texto = round(time.perf_counter() - tiempo_inicio_extraccion, 4)

        # Medir procesamiento
        tiempo_inicio_procesamiento = time.perf_counter()
        
        arxiv_id = pdf_path.stem
        raw_path = save_raw_text(arxiv_id, text, raw_dir)
        abstract, method = extract_abstract_from_text(text)
        abstract = clean_text(abstract)
        title = extract_title_from_text(text)
        title = clean_text(title)
        
        tiempo_procesamiento = round(time.perf_counter() - tiempo_inicio_procesamiento, 4)
        tiempo_total = round(tiempo_extraccion_texto + tiempo_procesamiento, 4)
        
        return {
            # ... otros campos ...
            "timing": {
                "extraccion_texto_seg": tiempo_extraccion_texto,
                "procesamiento_seg": tiempo_procesamiento,
                "total_seg": tiempo_total,
            },
        }
```

### 9.3 Campos de Timing en JSON

Cada registro en `extracted_pymupdf.json` y `extracted_docling.json` incluye:

**Estructura:**

```json
{
  "arxiv_id": "2301.07041",
  "category": "cs.AI",
  "title": "Deep Learning Architecture",
  "abstract_pymupdf": "...",
  "extraction_method": "pattern_based",
  "status": "success",
  "timing": {
    "extraccion_texto_seg": 0.0432,
    "procesamiento_seg": 0.0125,
    "total_seg": 0.0557
  }
}
```

| Campo | Tipo | Descripción | Rango Típico |
|-------|------|-----------|--------------|
| `extraccion_texto_seg` | float | Tiempo de lectura del PDF | 0.01 - 2.0s |
| `procesamiento_seg` | float | Tiempo de extracción/limpieza | 0.001 - 0.1s |
| `total_seg` | float | Suma de ambos tiempos | 0.01 - 2.1s |

**Notas:**
- Todos los valores están redondeados a **4 decimales**
- Incluidos también en registros **fallidos** (valores = 0.0 si no completaron)
- La medición es **por PDF**, no acumulativa

### 9.4 Reporte de Timing

Al finalizar cada script, se genera un archivo de resumen:

**Ubicación:**
- `reports/timing_pymupdf.json` (para extract_pymupdf.py)
- `reports/timing_docling.json` (para extract_docling.py)

**Estructura completa:**

```json
{
  "total_pdfs": 2000,
  "tiempo_total_seg": 320.4523,
  "tiempo_promedio_seg": 0.1602,
  "tiempo_mediano_seg": 0.1378,
  "tiempo_min_seg": 0.0812,
  "tiempo_max_seg": 4.3201,
  "percentil_95_seg": 0.4121,
  "por_categoria": {
    "cs.AI": {
      "promedio_seg": 0.1542,
      "total_seg": 30.2340
    },
    "cs.LG": {
      "promedio_seg": 0.1934,
      "total_seg": 38.6840
    },
    "cs.CV": {
      "promedio_seg": 0.1821,
      "total_seg": 45.5241
    }
  }
}
```

| Campo | Descripción | Interpretación |
|-------|----------|---|
| `total_pdfs` | Cantidad de PDFs procesados | Incluye éxitos y fallos |
| `tiempo_total_seg` | Suma de todos los tiempos | Tiempo total de ejecución |
| `tiempo_promedio_seg` | Media aritmética | Referencia de rendimiento típico |
| `tiempo_mediano_seg` | Percentil 50 | Menos sensible a outliers |
| `tiempo_min_seg` | PDFs más rápidos | PDFs simples/pequeños |
| `tiempo_max_seg` | PDFs más lentos | PDFs complejos/grandes |
| `percentil_95_seg` | P95 de la distribución | 95% de PDFs ≤ este tiempo |
| `por_categoria` | Estadísticas por categoría | Identifica categorías lentas |

### 9.5 Interpretación de Resultados

#### **Comparación PyMuPDF vs Docling**

**Tiempos esperados (arXiv típico):**

| Métrica | PyMuPDF | Docling |
|---------|---------|---------|
| Promedio | 0.06 - 0.12s | 0.3 - 0.8s |
| P95 | 0.20s | 1.2s |
| Max | 0.5s | 3.0s |
| Razón | 1x | ~5x más lento |

**Por qué Docling es más lento:**
- Conversion a markdown (análisis de layout)
- Búsqueda de encabezados (iteración de líneas)
- Validaciones múltiples (regex + estructura + longitud)
- Inicialización de modelos (primera ejecución)

#### **Análisis por Categoría**

Comparar tiempos entre categorías:

```python
import json

timing = json.load(open("reports/timing_docling.json"))

# Identificar categorías lentas
por_cat = timing["por_categoria"]
for cat, stats in sorted(por_cat.items(), 
                         key=lambda x: x[1]["promedio_seg"], 
                         reverse=True):
    print(f"{cat}: {stats['promedio_seg']:.4f}s (n={int(stats['total_seg']/stats['promedio_seg'])})")

# Ejemplo de salida:
# cs.LG: 0.1934s (n=200)
# cs.AI: 0.1542s (n=196)
# cs.CV: 0.1821s (n=190)
```

#### **Detección de Problemas**

**Señal de alerta: P95 > 1 segundo**

Indica que el 5% de PDFs toman mucho más tiempo (posibles PDFs:
- Muy grandes (100+ páginas)
- Complejos (tablas, figuras)
- Escaneados sin OCR bien configurado

```python
timing = json.load(open("reports/timing_docling.json"))

if timing["percentil_95_seg"] > 1.0:
    print("⚠️  P95 muy alto - revisar PDFs problemáticos")
    print(f"   Max: {timing['tiempo_max_seg']}s")
    print(f"   P95: {timing['percentil_95_seg']}s")
```

### 9.6 Casos de Uso

#### **Optimización de Pipeline**

Identificar cuello de botella:

```python
# Comparar componentes en PyMuPDF
timing_pymupdf = json.load(open("reports/timing_pymupdf.json"))
timing_docling = json.load(open("reports/timing_docling.json"))

# Supongamos que docling es más lento en conversion_seg
# → Considerar usar PyMuPDF + limpieza manual
# → O agregar paralelismo (multiprocessing)
```

#### **Evaluación de Hardware**

Comparar performance entre máquinas:

```bash
# En máquina lenta (CPU vieja)
# → timing_docling.json: promedio = 1.2s

# En máquina rápida (GPU NVIDIA)
# → timing_docling.json: promedio = 0.4s
# → Mejora: 3x más rápido
```

#### **Presupuesto de Tiempo**

Estimar tiempo total de procesamiento:

```python
timing = json.load(open("reports/timing_docling.json"))

total_pdfs_restantes = 5000  # PDFs no procesados
promedio_seg = timing["tiempo_promedio_seg"]

tiempo_estimado_seg = total_pdfs_restantes * promedio_seg
tiempo_estimado_horas = tiempo_estimado_seg / 3600

print(f"Estimado: {tiempo_estimado_horas:.1f} horas")
```

### 9.7 Evolución en el Tiempo

Cada ejecución sobrescribe `timing_*.json`, pero se pueden archivar:

```bash
# Backup antes de nueva ejecución
cp reports/timing_docling.json \
   reports/timing_docling_2025-05-13.json

# Luego ejecutar script
uv run python src/extraction/extract_docling.py

# Comparar evolución
git diff reports/timing_docling_2025-05-13.json reports/timing_docling.json
```

**Métrica a monitorear:** `tiempo_promedio_seg`

- ↑ Aumento = PDFs más complejos o hardware más lento
- ↓ Disminución = Optimizaciones en código o hardware más rápido

---

## 10. Referencias

- **PyMuPDF (fitz)**: https://pymupdf.io/
- **Docling**: https://github.com/DS4SD/docling
- **Regex Python**: https://docs.python.org/3/library/re.html
- **arXiv API**: https://arxiv.org/help/api
- **Python re module**: https://docs.python.org/3/library/re.html#re.search
- **Docling Documentation**: https://github.com/DS4SD/docling
- **time.perf_counter()**: https://docs.python.org/3/library/time.html#time.perf_counter

