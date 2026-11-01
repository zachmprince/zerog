"""Locate nuclide files through an OpenMC cross_sections.xml."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path


def cross_sections_path():
    """Path to the library's cross_sections.xml, from OPENMC_CROSS_SECTIONS."""
    try:
        return Path(os.environ["OPENMC_CROSS_SECTIONS"])
    except KeyError:
        raise RuntimeError("OPENMC_CROSS_SECTIONS is not set") from None


def nuclide_path(name, data_type="neutron"):
    """Path to the HDF5 file of a material, e.g. 'U235' or ('c_H_in_H2O', 'thermal')."""
    xml_path = cross_sections_path()
    root = ET.parse(xml_path).getroot()
    directory = root.findtext("directory")
    base = xml_path.parent / directory if directory else xml_path.parent
    for library in root.findall("library"):
        if library.get("type") == data_type and name in library.get("materials", "").split():
            return base / library.get("path")
    raise KeyError(f"{name} ({data_type}) is not in {xml_path}")
