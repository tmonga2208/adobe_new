 PDF Outline Extractor

## Overview
This solution extracts structured outlines from PDF documents, identifying titles and hierarchical headings (H1, H2, H3) with their page numbers.

## Key Features

### Multi-Library Support
- **Primary**: py-pdf-parser (as requested)
- **Fallback**: pdfplumber for broader compatibility
- Automatic fallback if py-pdf-parser isn't available

### Intelligent Heading Detection
- **Pattern Recognition**: Detects numbered sections (1., 1.1, 1.1.1), chapters, bullet points
- **Font Analysis**: Analyzes font sizes and styles to identify hierarchy
- **Positional Clues**: Considers text positioning, formatting (bold, caps, colons)
- **Content Filtering**: Removes noise and obvious non-headings

### Robust Processing
- Handles up to 50 pages per document
- Processes all PDFs in input directory automatically
- Graceful error handling with fallback outputs
- Optimized for performance (< 10 seconds for 50-page PDFs)

## Build Instructions

```bash
# Build the Docker image
docker build --platform linux/amd64 -t pdf-outline-extractor:v1.0 .
```

## Run Instructions

```bash
# Run the container
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-outline-extractor:v1.0
```

## Directory Structure
```
project/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── input/              # Place PDF files here
└── output/             # JSON outputs generated here
```

## Input/Output Format

### Input
- PDF files in `/app/input/` directory
- Maximum 50 pages per PDF

### Output
- JSON files in `/app/output/` directory
- Format: `filename.json` for each `filename.pdf`

### JSON Schema
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Main Section",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "Subsection",
      "page": 2
    },
    {
      "level": "H3",
      "text": "Sub-subsection", 
      "page": 3
    }
  ]
}
```

## Algorithm Details

### Title Detection
1. Analyzes first 20 text elements
2. Scores based on:
   - Position (earlier = better)
   - Font size (larger = better)
   - Bold formatting
   - Content patterns (avoids obvious headings)
   - Length (10-100 characters preferred)

### Heading Level Assignment
1. **Pattern-based**: Numbered sections get appropriate levels
2. **Font-based**: Larger fonts become higher-level headings
3. **Style-based**: Bold text and positioning clues
4. **Hierarchical**: H1 → H2 → H3 based on font size descending

### Performance Optimizations
- Efficient character grouping
- Font analysis caching
- Pattern matching optimization
- Memory-efficient processing

## Constraints Met
- ✅ Execution time: < 10 seconds for 50-page PDFs
- ✅ Model size: < 200MB (no ML models used)
- ✅ Network: No internet access required
- ✅ Runtime: CPU-only, AMD64 compatible
- ✅ Memory: Optimized for 16GB RAM systems

## Testing Recommendations
1. Test with simple PDFs (clear headings, consistent fonts)
2. Test with complex PDFs (mixed fonts, irregular formatting)
3. Test with academic papers, reports, manuals
4. Verify performance with 50-page documents
5. Check edge cases (no headings, all headings, mixed content)
