"""Utilities for loading and normalizing Picbreeder XML genomes."""

import zipfile
import xml.etree.ElementTree as ET


def _xml_to_dict(element):
    """Convert an XML element (and children) to a nested dictionary.

    Args:
        element: XML element from `xml.etree`.

    Returns:
        dict mapping XML structure to Python primitives.
    """
    node = {}
    if element.attrib:
        node.update({f"@{key}": value for key, value in element.attrib.items()})
    # end if
    children = list(element)
    if children:
        child_dict = {}
        for child in children:
            child_name = child.tag
            child_dict.setdefault(child_name, []).append(_xml_to_dict(child))
        # end for
        for key, value in child_dict.items():
            node[key] = value if len(value) > 1 else value[0]
        # end for
    else:
        if element.text and element.text.strip():
            node["#text"] = element.text.strip()
        # end if
    # end if
    return node
# end def _xml_to_dict

def load_zip_xml_as_dict(zip_file_path):
    """Load a Picbreeder genome zip and return a dictionary description.

    Args:
        zip_file_path: Path to `.zip` containing a single XML genome.

    Returns:
        dict with at least the `genome` key mirroring the XML tree.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        assert len(file_list) == 1
        for file_name in file_list:
            with zip_ref.open(file_name) as file:
                file_content = file.read().decode('utf-8')
            # end with
        # end for
    # end with
    element = ET.fromstring(file_content)
    root = _xml_to_dict(element)
    if 'genome' not in root:
        root = dict(genome=root)
    # end if
    return root
# end def load_zip_xml_as_dict
