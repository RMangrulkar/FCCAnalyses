import pdg
from lxml import etree

# Connect to PDG database
api = pdg.connect()

def pdg_name(pdgid):
    """Return PDG particle name using pdg API."""
    try:
        p = api.get_particle_by_mcid(int(pdgid))
        return p.name
    except Exception:
        return f"UNKNOWN({pdgid})"


#Load Pythia XML file
def load_xml(path):
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(path, parser)
    return tree.getroot()


#Extract decays for specific particle PDG IDs
def extract_decays(xml_root, pdg_list):
    decays = {}

    for p in xml_root.xpath("//particle"):
        pid = int(p.get("id"))
        if pid not in pdg_list:
            continue

        particle_name = p.get("name")

        channels = []
        for ch in p.xpath("channel"):
            br = float(ch.get("bRatio"))
            products = list(map(int, ch.get("products").split()))
            channels.append((br, products))

        decays[pid] = {
            "name": particle_name,
            "channels": channels
        }

    return decays


#Convert PDG IDs to Names using PDG API
def replace_with_names(decays):
    pretty = {}

    for pid, info in decays.items():
        pname = pdg_name(pid)

        named_channels = []
        for br, prods in info["channels"]:
            prod_names = [pdg_name(p) for p in prods]
            named_channels.append((br, prod_names))

        pretty[pid] = {
            "name": pname,
            "channels": named_channels
        }

    return pretty
       
    

if __name__ == "__main__":
    # B+ (521), Bc+ (541), B0 (511),tau (15)
    pdg_ids = [521, 541, 511, 15]

    xml_filepath = "pythia_decays/Pythia_ParticleData.xml"

    xml = load_xml(xml_filepath)
    raw_decays = extract_decays(xml, pdg_ids)
    pretty = replace_with_names(raw_decays)

    filename="pythia_decays/pythia_decay_channels.txt"
    with open(filename, "w") as f:
        for pid, info in pretty.items():
            print(f"\n===== {info['name']} (PDG {pid}) =====")
            f.write(f"===== {info['name']} (PDG {pid}) =====\n")
            for br, products in info["channels"]:
                print(f"  BR = {br:.6f} → {' + '.join(products)}")
                f.write(f"  BR = {br:.6f} → {' + '.join(products)}\n")
            f.write("\n")
            print(f"Saved decay table to {filename}")