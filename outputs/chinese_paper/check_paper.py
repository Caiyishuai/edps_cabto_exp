"""Validate the compiled manuscript against its extracted numerical evidence.
PDF checks use PyMuPDF; source-data hash verification is optional with project root.
"""
from pathlib import Path
import hashlib,json,re,sys
import pymupdf
from PIL import Image,ImageOps,ImageDraw
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    pdf=HERE/"paper.pdf";source=(HERE/"main.tex").read_text();ev=json.loads((HERE/"evidence.json").read_text());log=(HERE/"build/main.log").read_text(errors="replace")
    assert pdf.is_file() and pdf.read_bytes().startswith(b"%PDF-")
    bad=[line for line in log.splitlines() if any(x in line for x in ["undefined references","Citation '","Overfull ","Missing character","! LaTeX Error","! Package"])]
    assert not bad,bad
    assert ev["totals"]=={"goals":9,"actions":43,"episodes":17,"transitions":113}
    assert sum(r["old_full_matches"] for r in ev["paired_training_witnesses"])==21
    assert sum(r["witnesses"] for r in ev["paired_training_witnesses"])==31
    assert len(ev["model_patches"])==10
    for p in ev["model_patches"]:
        m=p["model"];assert (set(p["before"])|set(m["add"]))-set(m["del"])==set(p["after"])
    assert sum(r["returned"] for r in ev["camera_point_counts"])==18
    assert sum(r["abstained"] for r in ev["camera_point_counts"])==10
    cites=[]
    for group in re.findall(r"\\cite\{([^}]+)\}",source):cites.extend(group.split(","))
    bib=(HERE/"references.bib").read_text();keys=re.findall(r"@\w+\{([^,]+),",bib)
    assert len(keys)==14 and set(cites)<=set(keys)
    bbl=(HERE/"build/main.bbl").read_text();assert len(re.findall(r"\\entry\{",bbl))==14
    imagefiles=re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",source)
    for f in imagefiles:
        assert (HERE/f).is_file() or (HERE/"assets"/f).is_file(),f
    document=pymupdf.open(pdf);texts=[p.get_text() for p in document];text="\n".join(texts)
    assert len(document)>=10 and all(len(t.strip())>100 for t in texts)
    assert not re.search(r"\?\?|\ufffd",text)
    assert "关键词" in texts[0] and "Key words" in texts[0]
    assert "引言" in texts[1] and "参考文献" in text
    # Check visible text bounding boxes against page bounds, allowing normal margins.
    outside=[]
    for i,p in enumerate(document):
        for b in p.get_text("blocks"):
            if b[0]<-1 or b[1]<-1 or b[2]>p.rect.width+1 or b[3]>p.rect.height+1:outside.append(i+1)
    assert not outside,outside
    root=Path(sys.argv[1]) if len(sys.argv)>1 else None
    if root:
        for record in json.loads((HERE/"source_manifest.json").read_text()):assert sha(root/record["path"])==record["sha256"],record["path"]
    thumbnails=[]
    for i,p in enumerate(document):
        image=Image.frombytes("RGB",[p.get_pixmap(matrix=pymupdf.Matrix(.55,.55)).width,p.get_pixmap(matrix=pymupdf.Matrix(.55,.55)).height],p.get_pixmap(matrix=pymupdf.Matrix(.55,.55)).samples)
        image.thumbnail((230,330));canvas=Image.new("RGB",(250,360),"#e9e9e9");canvas.paste(image,((250-image.width)//2,8));ImageDraw.Draw(canvas).text((12,340),f"Page {i+1}",fill="#222222");thumbnails.append(canvas)
    contact=Image.new("RGB",(1000,360*((len(thumbnails)+3)//4)),"#d8d8d8")
    for i,img in enumerate(thumbnails):contact.paste(img,((i%4)*250,(i//4)*360))
    contact.save(HERE/"build/page_contact_sheet.png")
    result={"passed":True,"pdf_pages":len(document),"pdf_sha256":sha(pdf),"pdf_bytes":pdf.stat().st_size,"bibliography_entries":14,
       "figures":len(re.findall(r"\\begin\{figure\}",source)),"tables":len(re.findall(r"\\begin\{(?:table|longtable)\}",source)),"algorithms":len(re.findall(r"\\begin\{algorithm\}",source)),
       "undefined_citations_or_references":False,"overfull_boxes":False,"missing_characters":False,"source_hashes_rechecked":root is not None,
       "main_counts":ev["totals"],"new_simulation_runs_performed_for_manuscript":False,
       "scope":"Chinese empirical research manuscript; selected frozen results, dual oracle+weld and perception limitations explicitly reported"}
    (HERE/"quality_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
