from tokrag.parse.extract import extract_sections_from_html, extract_sections_from_pdf


def test_html_extracts_headings_and_paragraphs():
    html = """
    <html><body><article>
      <h2>1. Introduction</h2>
      <p>This paper studies tokenization.</p>
      <h2>2. Method</h2>
      <p>We propose a new subword algorithm.</p>
      <figcaption>Figure 1: an overview diagram.</figcaption>
    </article></body></html>
    """
    sections = extract_sections_from_html(html)
    assert sections[0] == ("Introduction", "This paper studies tokenization.")
    assert sections[1][0] == "Method"
    assert "subword algorithm" in sections[1][1]
    assert "overview diagram" in sections[1][1]


def test_html_drops_references_section():
    html = """
    <html><body><article>
      <h2>Introduction</h2>
      <p>Body text here.</p>
      <h2>References</h2>
      <p>Smith et al. 2020. Some Other Paper.</p>
    </article></body></html>
    """
    sections = extract_sections_from_html(html)
    assert len(sections) == 1
    assert sections[0][0] == "Introduction"
    assert "Smith" not in sections[0][1]


def test_html_with_no_headings_falls_back_to_abstract_bucket():
    html = "<html><body><article><p>Just one paragraph, no headings.</p></article></body></html>"
    sections = extract_sections_from_html(html)
    assert len(sections) == 1
    assert sections[0][0] == "Abstract"


def test_pdf_strips_repeated_running_header(tmp_path):
    import fitz

    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        page.insert_text((72, 40), "Proceedings of ACL 2024, pages 1-10", fontsize=9)
        page.insert_text((72, 72), f"1. Introduction\nBody text on page {i}.", fontsize=11)
    pdf_path = tmp_path / "running_header.pdf"
    doc.save(pdf_path)
    doc.close()

    from tokrag.parse.extract import extract_sections_from_pdf

    sections = extract_sections_from_pdf(pdf_path)
    full_text = " ".join(t for _, t in sections)
    assert "Proceedings of ACL 2024" not in full_text


def test_pdf_detects_numbered_heading_pattern(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1 Introduction\nBody text here.\n4.2 Results\nMore body text.", fontsize=11)
    pdf_path = tmp_path / "numbered.pdf"
    doc.save(pdf_path)
    doc.close()

    from tokrag.parse.extract import extract_sections_from_pdf

    sections = extract_sections_from_pdf(pdf_path)
    headings = [h for h, _ in sections]
    assert "Introduction" in headings
    assert "Results" in headings


def test_pdf_extraction_splits_on_recognized_headers(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    body = "1. Introduction\nThis paper studies tokenization.\n2. Method\nWe propose a new algorithm.\nReferences\nSmith et al. 2020."
    page.insert_text((72, 72), body, fontsize=11)
    pdf_path = tmp_path / "fake.pdf"
    doc.save(pdf_path)
    doc.close()

    sections = extract_sections_from_pdf(pdf_path)
    headings = [h for h, _ in sections]
    assert "Introduction" in headings
    assert "Method" in headings
    assert not any("Smith" in t for _, t in sections)
