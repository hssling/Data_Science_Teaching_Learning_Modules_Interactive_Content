#!/usr/bin/env python3
"""
Documentation Compiler
======================

Compiles the comprehensive data science curriculum documentation into various formats:
- PDF (with table of contents and indexing)
- HTML (interactive with navigation)
- Markdown (consolidated reference)

Usage:
    python compile_documentation.py --format pdf
    python compile_documentation.py --format html
    python compile_documentation.py --format all
"""

import os
import sys
from pathlib import Path
import markdown
import pdfkit
from bs4 import BeautifulSoup
import json
from datetime import datetime
import argparse
import shutil

class DocumentationCompiler:
    """Compiles curriculum documentation into various formats"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.output_dir = self.project_root / "compiled_docs"
        self.output_dir.mkdir(exist_ok=True)

        # Documentation structure
        self.doc_structure = {
            "README.md": "Main README",
            "CURRICULUM_GUIDE.md": "Complete Curriculum Guide",
            "CURRICULUM_SUMMARY.md": "Curriculum Summary",
            "modules/": "Module Documentation",
            "projects/README.md": "Projects Documentation",
            "resources/learning_resources.md": "Learning Resources",
            "quizzes/": "Quiz Documentation",
            "exercises/": "Exercise Documentation"
        }

    def collect_all_markdown(self):
        """Collect all markdown files in the curriculum"""
        markdown_files = []

        # Main documentation files
        main_files = [
            "README.md",
            "CURRICULUM_GUIDE.md",
            "CURRICULUM_SUMMARY.md",
            "CURRICULUM_OVERVIEW.md",
            "resources/learning_resources.md",
            "projects/README.md"
        ]

        for file_path in main_files:
            if (self.project_root / file_path).exists():
                try:
                    content = (self.project_root / file_path).read_text(encoding='utf-8', errors='replace')
                    markdown_files.append({
                        'path': file_path,
                        'content': content,
                        'title': self.get_file_title(file_path)
                    })
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
                    continue

        # Module documentation
        modules_dir = self.project_root / "modules"
        if modules_dir.exists():
            for module_dir in sorted(modules_dir.iterdir()):
                if module_dir.is_dir():
                    readme_file = module_dir / "README.md"
                    if readme_file.exists():
                        try:
                            content = readme_file.read_text(encoding='utf-8', errors='replace')
                            markdown_files.append({
                                'path': str(readme_file.relative_to(self.project_root)),
                                'content': content,
                                'title': f"Module: {module_dir.name.replace('_', ' ').title()}"
                            })
                        except Exception as e:
                            print(f"Warning: Could not read {readme_file}: {e}")
                            continue

        return markdown_files

    def get_file_title(self, file_path):
        """Extract title from markdown file"""
        try:
            content = (self.project_root / file_path).read_text()
            lines = content.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                if line.startswith('# '):
                    return line[2:].strip()
            return file_path.replace('.md', '').replace('_', ' ').title()
        except:
            return file_path

    def create_consolidated_markdown(self):
        """Create a consolidated markdown file"""
        print("📝 Creating consolidated markdown documentation...")

        markdown_files = self.collect_all_markdown()

        # Create consolidated content
        consolidated = f"""# 📚 Comprehensive Data Science Curriculum - Complete Documentation

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Files:** {len(markdown_files)}

---

"""

        for i, file_info in enumerate(markdown_files, 1):
            consolidated += f"""## {i}. {file_info['title']}

**File:** `{file_info['path']}`

---

{file_info['content']}

---

"""

        # Save consolidated markdown
        output_file = self.output_dir / "curriculum_complete.md"
        output_file.write_text(consolidated)

        print(f"✅ Consolidated markdown saved: {output_file}")
        return str(output_file)

    def create_html_documentation(self):
        """Create interactive HTML documentation"""
        print("🌐 Creating interactive HTML documentation...")

        markdown_files = self.collect_all_markdown()

        # HTML template
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📚 Data Science Curriculum - Complete Documentation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .navigation {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .nav-links {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .nav-link {
            background: #f0f0f0;
            padding: 8px 16px;
            border-radius: 20px;
            text-decoration: none;
            color: #333;
            transition: all 0.3s ease;
        }
        .nav-link:hover {
            background: #667eea;
            color: white;
        }
        .content-section {
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .section-header {
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .file-info {
            background: #f8f9fa;
            padding: 10px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
            font-family: monospace;
        }
        .code-block {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            overflow-x: auto;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .search-box {
            width: 100%;
            padding: 12px;
            border: 2px solid #667eea;
            border-radius: 25px;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .module-filter {
            margin-bottom: 20px;
        }
        .filter-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .filter-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .filter-btn:hover {
            background: #5a67d8;
        }
        .filter-btn.active {
            background: #4c51bf;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Comprehensive Data Science Curriculum</h1>
        <p>Complete Documentation - Interactive HTML Version</p>
        <p><strong>Author:</strong> Dr. Siddalingaiah H S, Professor, Community Medicine</p>
        <p><strong>Institution:</strong> Shridevi Institute of Medical Sciences and Research Hospital, Tumkur</p>
        <p><strong>Contact:</strong> hssling@yahoo.com | 8941087719</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="navigation">
        <h2>🧭 Quick Navigation</h2>
        <input type="text" class="search-box" id="searchBox" placeholder="Search documentation...">
        <div class="module-filter">
            <div class="filter-buttons">
                <button class="filter-btn active" onclick="filterContent('all')">All</button>
                <button class="filter-btn" onclick="filterContent('modules')">Modules</button>
                <button class="filter-btn" onclick="filterContent('projects')">Projects</button>
                <button class="filter-btn" onclick="filterContent('resources')">Resources</button>
                <button class="filter-btn" onclick="filterContent('quizzes')">Quizzes</button>
            </div>
        </div>
        <div class="nav-links">
"""

        # Add navigation links
        for i, file_info in enumerate(markdown_files, 1):
            html_template += f'            <a href="#section-{i}" class="nav-link">{i}. {file_info["title"]}</a>\n'

        html_template += """        </div>
    </div>
"""

        # Add content sections
        for i, file_info in enumerate(markdown_files, 1):
            # Convert markdown to HTML
            html_content = markdown.markdown(file_info['content'], extensions=['tables', 'fenced_code'])

            html_template += f"""
    <div class="content-section" id="section-{i}" data-category="{self.get_category(file_info['path'])}">
        <h2 class="section-header">{i}. {file_info['title']}</h2>
        <div class="file-info">
            📄 File: {file_info['path']}
        </div>
        <div class="markdown-content">
            {html_content}
        </div>
    </div>
"""

        # Add footer and JavaScript
        html_template += """
    <div class="footer">
        <h3>🎉 Thank You for Using the Data Science Curriculum!</h3>
        <p>This comprehensive curriculum is designed to transform beginners into industry-ready data scientists.</p>
        <p><strong>Built with ❤️ for the data science community</strong></p>
        <p>© 2024 Comprehensive Data Science Curriculum</p>
    </div>

    <script>
        // Search functionality
        document.getElementById('searchBox').addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const sections = document.querySelectorAll('.content-section');

            sections.forEach(section => {
                const text = section.textContent.toLowerCase();
                if (text.includes(searchTerm) || searchTerm === '') {
                    section.style.display = 'block';
                } else {
                    section.style.display = 'none';
                }
            });
        });

        // Filter functionality
        function filterContent(category) {
            const sections = document.querySelectorAll('.content-section');
            const buttons = document.querySelectorAll('.filter-btn');

            // Update button states
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            // Filter sections
            sections.forEach(section => {
                if (category === 'all' || section.dataset.category === category) {
                    section.style.display = 'block';
                } else {
                    section.style.display = 'none';
                }
            });
        }

        // Smooth scrolling for navigation links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = this.getAttribute('href');
                const targetSection = document.querySelector(targetId);
                targetSection.scrollIntoView({ behavior: 'smooth' });
            });
        });
    </script>
</body>
</html>"""

        # Save HTML file
        output_file = self.output_dir / "curriculum_complete.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)

        print(f"✅ Interactive HTML documentation saved: {output_file}")
        return str(output_file)

    def get_category(self, file_path):
        """Get category for filtering"""
        if 'modules/' in file_path:
            return 'modules'
        elif 'projects/' in file_path:
            return 'projects'
        elif 'resources/' in file_path:
            return 'resources'
        elif 'quizzes/' in file_path:
            return 'quizzes'
        else:
            return 'general'

    def create_pdf_optimized_html(self):
        """Create PDF-optimized HTML with compact spacing"""
        print("📄 Creating PDF-optimized HTML template...")

        markdown_files = self.collect_all_markdown()

        # Compact HTML template for PDF
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Data Science Curriculum - PDF Version</title>
    <style>
        @media print {{
            body {{ margin: 0; }}
            .page-break {{ page-break-before: always; }}
        }}
        body {{
            font-family: 'Times New Roman', serif;
            font-size: 12px;
            line-height: 1.4;
            color: #000;
            margin: 0;
            padding: 15px;
            max-width: none;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #000;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .title {{ font-size: 18px; font-weight: bold; margin-bottom: 5px; }}
        .subtitle {{ font-size: 14px; margin-bottom: 5px; }}
        .author-info {{ font-size: 11px; margin-bottom: 10px; }}
        .toc {{
            margin-bottom: 15px;
        }}
        .toc-title {{
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 8px;
            text-decoration: underline;
        }}
        .toc-item {{
            margin-bottom: 3px;
            font-size: 11px;
        }}
        .section {{
            margin-bottom: 12px;
        }}
        .section-title {{
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 5px;
            border-bottom: 1px solid #666;
            padding-bottom: 3px;
        }}
        .file-info {{
            font-size: 10px;
            color: #666;
            margin-bottom: 8px;
            font-style: italic;
        }}
        .content {{
            margin-bottom: 8px;
        }}
        .content p {{
            margin: 4px 0;
        }}
        .content ul, .content ol {{
            margin: 4px 0;
            padding-left: 20px;
        }}
        .content li {{
            margin: 2px 0;
        }}
        .content h1, .content h2, .content h3,
        .content h4, .content h5, .content h6 {{
            margin: 8px 0 4px 0;
            font-size: 13px;
            font-weight: bold;
        }}
        .content h1 {{ font-size: 15px; }}
        .content h2 {{ font-size: 14px; }}
        .content pre {{
            background: #f5f5f5;
            padding: 5px;
            margin: 5px 0;
            font-size: 10px;
            font-family: 'Courier New', monospace;
            overflow-wrap: break-word;
            white-space: pre-wrap;
        }}
        .content code {{
            font-family: 'Courier New', monospace;
            font-size: 11px;
            background: #f0f0f0;
            padding: 1px 3px;
        }}
        .content table {{
            border-collapse: collapse;
            margin: 5px 0;
            font-size: 11px;
        }}
        .content th, .content td {{
            border: 1px solid #666;
            padding: 3px 5px;
            text-align: left;
        }}
        .content th {{
            background: #e0e0e0;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">📚 Comprehensive Data Science Curriculum</div>
        <div class="subtitle">Complete Documentation - PDF Version</div>
        <div class="author-info">
            <strong>Author:</strong> Dr. Siddalingaiah H S, Professor, Community Medicine<br>
            <strong>Institution:</strong> Shridevi Institute of Medical Sciences and Research Hospital, Tumkur<br>
            <strong>Contact:</strong> hssling@yahoo.com | 8941087719<br>
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>

    <div class="toc">
        <div class="toc-title">Table of Contents</div>
"""

        # Add table of contents
        for i, file_info in enumerate(markdown_files, 1):
            html_template += f'        <div class="toc-item">{i}. {file_info["title"]}</div>\n'

        html_template += "    </div>\n"

        # Add content sections
        for i, file_info in enumerate(markdown_files, 1):
            # Convert markdown to HTML
            html_content = markdown.markdown(file_info['content'], extensions=['tables', 'fenced_code'])

            html_template += f"""
    <div class="section">
        <div class="section-title">{i}. {file_info['title']}</div>
        <div class="file-info">File: {file_info['path']}</div>
        <div class="content">
            {html_content}
        </div>
    </div>
"""

        html_template += """
</body>
</html>"""

        # Save PDF-optimized HTML file
        pdf_html_file = self.output_dir / "curriculum_pdf_version.html"
        with open(pdf_html_file, 'w', encoding='utf-8') as f:
            f.write(html_template)

        print(f"✅ PDF-optimized HTML saved: {pdf_html_file}")
        return str(pdf_html_file)

    def create_pdf_documentation(self):
        """Create PDF documentation with table of contents"""
        print("📄 Creating PDF documentation...")

        try:
            # Try multiple PDF generation methods
            pdf_file = self.output_dir / "curriculum_complete.pdf"

            # Method 1: Try xhtml2pdf (already installed)
            try:
                print("🔄 Attempting PDF generation with xhtml2pdf...")
                from xhtml2pdf import pisa

                # Use PDF-optimized HTML
                html_file = self.create_pdf_optimized_html()
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Convert HTML to PDF
                with open(pdf_file, 'wb') as f:
                    pisa.CreatePDF(html_content, dest=f)

                if pdf_file.exists() and pdf_file.stat().st_size > 1000:  # Check if PDF was created successfully
                    print(f"✅ PDF documentation saved with xhtml2pdf: {pdf_file}")
                    return str(pdf_file)
                else:
                    raise Exception("PDF file not created or too small")

            except Exception as xhtml_error:
                print(f"⚠️  xhtml2pdf failed: {str(xhtml_error)}")

                # Method 2: Try wkhtmltopdf if available
                try:
                    print("🔄 Attempting PDF generation with wkhtmltopdf...")
                    # PDF options - optimized for compact layout
                    pdf_options = {
                        'page-size': 'A4',
                        'margin-top': '0.3in',
                        'margin-right': '0.4in',
                        'margin-bottom': '0.3in',
                        'margin-left': '0.4in',
                        'encoding': 'UTF-8',
                        'no-outline': None,
                        'enable-local-file-access': None,
                        'disable-smart-shrinking': None,
                        'print-media-type': None,
                        'dpi': 96,
                        'zoom': 1.0
                    }

                    # Use PDF-optimized HTML
                    html_file = self.create_pdf_optimized_html()
                    pdfkit.from_file(html_file, str(pdf_file), options=pdf_options)

                    print(f"✅ PDF documentation saved with wkhtmltopdf: {pdf_file}")
                    return str(pdf_file)

                except Exception as wkhtml_error:
                    print(f"⚠️  wkhtmltopdf not available: {str(wkhtml_error)}")

                    # Method 3: Create enhanced text-based PDF
                    print("📋 Creating enhanced text-based PDF solution...")
                    self.create_enhanced_pdf_alternative()
                    return None

        except Exception as e:
            print(f"❌ PDF generation failed: {str(e)}")
            print("💡 Alternative: Use browser's Print to PDF feature on the HTML file")
            print("   1. Open compiled_docs/curriculum_complete.html in your browser")
            print("   2. Press Ctrl+P (or Cmd+P on Mac)")
            print("   3. Select 'Save as PDF' or 'Print to PDF'")
            return None

    def create_text_pdf_alternative(self):
        """Create a text-based PDF alternative using reportlab"""
        print("📝 Creating text-based PDF alternative...")

        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.units import inch
            from reportlab.lib import colors

            # Create PDF document
            pdf_file = self.output_dir / "curriculum_text_version.pdf"
            doc = SimpleDocTemplate(str(pdf_file), pagesize=A4)
            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # Center alignment
                textColor=colors.darkblue
            )

            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=20,
                textColor=colors.darkgreen
            )

            normal_style = styles['Normal']

            # Collect content
            markdown_files = self.collect_all_markdown()
            story = []

            # Title page
            story.append(Paragraph("📚 Comprehensive Data Science Curriculum", title_style))
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph("Complete Documentation", heading_style))
            story.append(Spacer(1, 0.25*inch))
            story.append(Paragraph("Author: Dr. Siddalingaiah H S", normal_style))
            story.append(Paragraph("Professor, Community Medicine", normal_style))
            story.append(Paragraph("Shridevi Institute of Medical Sciences and Research Hospital, Tumkur", normal_style))
            story.append(Paragraph("Email: hssling@yahoo.com | Phone: 8941087719", normal_style))
            story.append(Spacer(1, 0.25*inch))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            story.append(Paragraph(f"Total Sections: {len(markdown_files)}", normal_style))
            story.append(PageBreak())

            # Table of contents
            story.append(Paragraph("Table of Contents", heading_style))
            story.append(Spacer(1, 0.25*inch))

            for i, file_info in enumerate(markdown_files, 1):
                toc_entry = f"{i}. {file_info['title']}"
                story.append(Paragraph(toc_entry, normal_style))
            story.append(PageBreak())

            # Content sections
            for i, file_info in enumerate(markdown_files, 1):
                story.append(Paragraph(f"{i}. {file_info['title']}", heading_style))
                story.append(Paragraph(f"File: {file_info['path']}", ParagraphStyle('FilePath', parent=normal_style, fontSize=10, textColor=colors.gray)))
                story.append(Spacer(1, 0.1*inch))

                # Convert markdown content to simple text
                content_lines = file_info['content'].split('\n')
                for line in content_lines[:50]:  # Limit content per section for PDF size
                    if line.strip():
                        # Basic markdown to text conversion
                        line = line.replace('#', '').replace('*', '').replace('_', '').strip()
                        if line:
                            story.append(Paragraph(line, normal_style))

                story.append(Spacer(1, 0.25*inch))
                story.append(PageBreak())

            # Build PDF
            doc.build(story)
            print(f"✅ Text-based PDF alternative saved: {pdf_file}")
            print("💡 Note: This is a simplified text version. For full formatting, use the HTML version with browser Print to PDF.")

        except ImportError:
            print("❌ ReportLab not available for PDF generation")
            print("💡 To create PDF manually:")
            print("   1. Open compiled_docs/curriculum_complete.html in your browser")
            print("   2. Press Ctrl+P (or Cmd+P on Mac)")
            print("   3. Select 'Save as PDF' or 'Print to PDF'")
        except Exception as e:
            print(f"❌ Text PDF creation failed: {str(e)}")
            print("💡 Alternative: Use browser's Print to PDF feature on the HTML file")

    def create_indexed_documentation(self):
        """Create indexed documentation with hyperlinked table of contents"""
        print("📚 Creating indexed documentation...")

        markdown_files = self.collect_all_markdown()

        # Create table of contents
        toc = "# 📖 Table of Contents\n\n"
        for i, file_info in enumerate(markdown_files, 1):
            toc += f"{i}. [{file_info['title']}](#section-{i})\n"

        # Create indexed content
        indexed_content = f"""# 📚 Comprehensive Data Science Curriculum - Indexed Documentation

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Sections:** {len(markdown_files)}

## Table of Contents

{toc}

---

"""

        for i, file_info in enumerate(markdown_files, 1):
            indexed_content += f"""## {i}. {file_info['title']}

**File:** `{file_info['path']}`

---

{file_info['content']}

---

[⬆️ Back to Table of Contents](#table-of-contents)

"""

        # Save indexed documentation
        output_file = self.output_dir / "curriculum_indexed.md"
        output_file.write_text(indexed_content)

        print(f"✅ Indexed documentation saved: {output_file}")
        return str(output_file)

    def create_curriculum_json(self):
        """Create JSON representation of curriculum structure"""
        print("📊 Creating curriculum JSON structure...")

        curriculum_structure = {
            "metadata": {
                "title": "Comprehensive Data Science Curriculum",
                "version": "1.0.0",
                "generated": datetime.now().isoformat(),
                "total_modules": 14,
                "total_quizzes": 3,
                "total_projects": 3,
                "total_exercises": 2,
                "author": {
                    "name": "Dr. Siddalingaiah H S",
                    "title": "Professor, Community Medicine",
                    "institution": "Shridevi Institute of Medical Sciences and Research Hospital, Tumkur",
                    "email": "hssling@yahoo.com",
                    "phone": "8941087719"
                }
            },
            "structure": {
                "phases": [
                    {
                        "name": "Foundations",
                        "duration": "2-3 months",
                        "modules": [1, 2, 3],
                        "focus": "Basics & Programming"
                    },
                    {
                        "name": "Data Engineering",
                        "duration": "2-3 months",
                        "modules": [4, 5, 6],
                        "focus": "Data Pipeline"
                    },
                    {
                        "name": "ML & Deep Learning",
                        "duration": "3-4 months",
                        "modules": [7, 8, 9],
                        "focus": "Models & Evaluation"
                    },
                    {
                        "name": "Production & Career",
                        "duration": "2-3 months",
                        "modules": [10, 11, 12, 13, 14],
                        "focus": "MLOps & Professional"
                    }
                ]
            },
            "modules": [],
            "resources": {
                "quizzes": [],
                "exercises": [],
                "projects": []
            }
        }

        # Add module details
        modules_dir = self.project_root / "modules"
        if modules_dir.exists():
            for module_dir in sorted(modules_dir.iterdir()):
                if module_dir.is_dir() and (module_dir.name.startswith('01_') or
                                          module_dir.name.startswith('02_') or
                                          module_dir.name.startswith('03_') or
                                          module_dir.name.startswith('04_') or
                                          module_dir.name.startswith('05_') or
                                          module_dir.name.startswith('06_') or
                                          module_dir.name.startswith('07_') or
                                          module_dir.name.startswith('08_') or
                                          module_dir.name.startswith('09_') or
                                          module_dir.name.startswith('10_') or
                                          module_dir.name.startswith('11_') or
                                          module_dir.name.startswith('12_') or
                                          module_dir.name.startswith('13_') or
                                          module_dir.name.startswith('14_')):
                    module_info = {
                        "id": int(module_dir.name.split('_')[0]),
                        "directory": module_dir.name,
                        "readme": str((module_dir / "README.md").relative_to(self.project_root)) if (module_dir / "README.md").exists() else None,
                        "examples": [str(f.relative_to(self.project_root)) for f in module_dir.glob("*.py")]
                    }
                    curriculum_structure["modules"].append(module_info)

        # Add resource details
        quizzes_dir = self.project_root / "quizzes"
        if quizzes_dir.exists():
            curriculum_structure["resources"]["quizzes"] = [str(f.relative_to(self.project_root)) for f in quizzes_dir.glob("*.py")]

        exercises_dir = self.project_root / "exercises"
        if exercises_dir.exists():
            curriculum_structure["resources"]["exercises"] = [str(f.relative_to(self.project_root)) for f in exercises_dir.glob("*.py")]

        projects_dir = self.project_root / "projects"
        if projects_dir.exists():
            curriculum_structure["resources"]["projects"] = [str(f.relative_to(self.project_root)) for f in projects_dir.glob("*.py")]

        # Save JSON
        json_file = self.output_dir / "curriculum_structure.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(curriculum_structure, f, indent=2, ensure_ascii=False)

        print(f"✅ Curriculum JSON structure saved: {json_file}")
        return str(json_file)

    def compile_all_formats(self):
        """Compile documentation in all available formats"""
        print("🚀 Compiling documentation in all formats...")
        print("=" * 60)

        results = {}

        # Create consolidated markdown
        try:
            results['markdown'] = self.create_consolidated_markdown()
        except Exception as e:
            print(f"❌ Markdown compilation failed: {str(e)}")

        # Create indexed documentation
        try:
            results['indexed'] = self.create_indexed_documentation()
        except Exception as e:
            print(f"❌ Indexed documentation failed: {str(e)}")

        # Create HTML documentation
        try:
            results['html'] = self.create_html_documentation()
        except Exception as e:
            print(f"❌ HTML compilation failed: {str(e)}")

        # Create PDF documentation
        try:
            results['pdf'] = self.create_pdf_documentation()
        except Exception as e:
            print(f"❌ PDF compilation failed: {str(e)}")

        # Create JSON structure
        try:
            results['json'] = self.create_curriculum_json()
        except Exception as e:
            print(f"❌ JSON creation failed: {str(e)}")

        # Summary
        print("\n" + "=" * 60)
        print("📊 COMPILATION SUMMARY")
        print("=" * 60)

        successful = 0
        for format_name, file_path in results.items():
            if file_path:
                print(f"✅ {format_name.upper()}: {file_path}")
                successful += 1
            else:
                print(f"❌ {format_name.upper()}: Failed")

        print(f"\nSuccessfully compiled: {successful}/{len(results)} formats")
        print(f"Output directory: {self.output_dir}")

        return results

    def show_usage_guide(self):
        """Show usage guide for compiled documentation"""
        print("\n" + "=" * 60)
        print("📖 DOCUMENTATION USAGE GUIDE")
        print("=" * 60)

        print("Available Formats:")
        print("• curriculum_complete.md     - Consolidated markdown")
        print("• curriculum_indexed.md      - Hyperlinked indexed markdown")
        print("• curriculum_complete.html   - Interactive HTML with navigation")
        print("• curriculum_complete.pdf    - Professional PDF (if wkhtmltopdf installed)")
        print("• curriculum_structure.json  - Machine-readable structure")

        print("\nRecommended Usage:")
        print("• HTML: Best for interactive browsing and navigation")
        print("• PDF: Best for printing and offline reading")
        print("• Markdown: Best for version control and editing")
        print("• JSON: Best for programmatic access and integration")

        print("\nHTML Features:")
        print("• Interactive table of contents")
        print("• Search functionality")
        print("• Category filtering (modules, projects, resources)")
        print("• Responsive design")

        print("\nPDF Features:")
        print("• Professional formatting")
        print("• Table of contents")
        print("• Print-optimized layout")
        print("• Bookmarks and navigation")

def main():
    parser = argparse.ArgumentParser(description='Compile Data Science Curriculum Documentation')
    parser.add_argument('--format', choices=['markdown', 'html', 'pdf', 'indexed', 'json', 'all'],
                       default='all', help='Output format (default: all)')
    parser.add_argument('--output-dir', default='compiled_docs',
                       help='Output directory (default: compiled_docs)')

    args = parser.parse_args()

    compiler = DocumentationCompiler()
    compiler.output_dir = Path(args.output_dir)
    compiler.output_dir.mkdir(exist_ok=True)

    if args.format == 'all':
        results = compiler.compile_all_formats()
        compiler.show_usage_guide()
    elif args.format == 'markdown':
        result = compiler.create_consolidated_markdown()
        print(f"✅ Markdown documentation: {result}")
    elif args.format == 'html':
        result = compiler.create_html_documentation()
        print(f"✅ HTML documentation: {result}")
    elif args.format == 'pdf':
        result = compiler.create_pdf_documentation()
        print(f"✅ PDF documentation: {result}")
    elif args.format == 'indexed':
        result = compiler.create_indexed_documentation()
        print(f"✅ Indexed documentation: {result}")
    elif args.format == 'json':
        result = compiler.create_curriculum_json()
        print(f"✅ JSON structure: {result}")

if __name__ == "__main__":
    main()
