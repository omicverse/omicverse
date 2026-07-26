"""
pdbfixer.py: Fixes problems in PDB files

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2013-2021 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import absolute_import
__author__ = "Peter Eastman"
__version__ = "1.7"

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmm.app.internal.pdbstructure import PdbStructure
from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader
from openmm.app.element import hydrogen, oxygen
from openmm.app.forcefield import NonbondedGenerator

# Support Cythonized functions in OpenMM 7.3
# and also implementations in older versions.
try:
    from openmm.app.internal import compiled
    matchResidue = compiled.matchResidueToTemplate
except ImportError:
    matchResidue = app.forcefield._matchResidue

import numpy as np
import numpy.linalg as lin
import sys
import os
import os.path
import math


if sys.version_info >= (3,0):
    from urllib.request import urlopen
    from io import StringIO
else:
    from urllib2 import urlopen
    from cStringIO import StringIO

substitutions = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}
proteinResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR', 'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL']
rnaResidues = ['A', 'G', 'C', 'U', 'I']
dnaResidues = ['DA', 'DG', 'DC', 'DT', 'DI']

class Sequence(object):
    """Sequence holds the sequence of a chain, as specified by SEQRES records."""
    def __init__(self, chainId, residues):
        self.chainId = chainId
        self.residues = residues

class ModifiedResidue(object):
    """ModifiedResidue holds information about a modified residue, as specified by a MODRES record."""
    def __init__(self, chainId, number, residueName, standardName):
        self.chainId = chainId
        self.number = number
        self.residueName = residueName
        self.standardName = standardName

def _guessFileFormat(file, filename):
    """Guess whether a file is PDB or PDBx/mmCIF based on its filename and contents."""
    filename = filename.lower()
    if '.pdbx' in filename or '.cif' in filename:
        return 'pdbx'
    if '.pdb' in filename:
        return 'pdb'
    for line in file:
        if line.startswith('data_') or line.startswith('loop_'):
            file.seek(0)
            return 'pdbx'
        if line.startswith('HEADER') or line.startswith('REMARK') or line.startswith('TITLE '):
            file.seek(0)
            return 'pdb'

    # It's certainly not a valid PDBx/mmCIF.  Guess that it's a PDB.

    file.seek(0)
    return 'pdb'

def _overlayPoints(points1, points2):
    """Given two sets of points, determine the translation and rotation that matches them as closely as possible.

    Parameters
    ----------
    points1 (numpy array of openmm.unit.Quantity with units compatible with distance) - reference set of coordinates
    points2 (numpy array of openmm.unit.Quantity with units compatible with distance) - set of coordinates to be rotated

    Returns
    -------
    translate2 - vector to translate points2 by in order to center it
    rotate - rotation matrix to apply to centered points2 to map it on to points1
    center1 - center of points1

    Notes
    -----
    This is based on W. Kabsch, Acta Cryst., A34, pp. 828-829 (1978).

    """

    if len(points1) == 0:
        return (mm.Vec3(0, 0, 0), np.identity(3), mm.Vec3(0, 0, 0))
    if len(points1) == 1:
        return (points1[0], np.identity(3), -1*points2[0])

    # Compute centroids.

    center1 = unit.sum(points1)/float(len(points1))
    center2 = unit.sum(points2)/float(len(points2))

    # Compute R matrix.

    R = np.zeros((3, 3))
    for p1, p2 in zip(points1, points2):
        x = p1-center1
        y = p2-center2
        for i in range(3):
            for j in range(3):
                R[i][j] += y[i]*x[j]

    # Use an SVD to compute the rotation matrix.

    (u, s, v) = lin.svd(R)
    return (-1*center2, np.dot(u, v).transpose(), center1)

def _findUnoccupiedDirection(point, positions):
    """Given a point in space and a list of atom positions, find the direction in which the local density of atoms is lowest."""

    point = point.value_in_unit(unit.nanometers)
    direction = mm.Vec3(0, 0, 0)
    for pos in positions.value_in_unit(unit.nanometers):
        delta = pos-point
        distance = unit.norm(delta)
        if distance > 0.1:
            distance2 = distance*distance
            direction -= delta/(distance2*distance2)
    direction /= unit.norm(direction)
    return direction

class PDBFixer(object):
    """PDBFixer implements many tools for fixing problems in PDB and PDBx/mmCIF files.
    """

    def __init__(self, filename=None, pdbfile=None, pdbxfile=None, url=None, pdbid=None):
        """Create a new PDBFixer instance to fix problems in a PDB or PDBx/mmCIF file.

        Parameters
        ----------
        filename : str, optional, default=None
            The name of the file to read.  The format is determined automatically based on the filename extension, or if
            that is ambiguous, by looking at the file content.
        pdbfile : file, optional, default=None
            A file-like object from which the PDB file is to be read.
            The file is not closed after reading.
        pdbxfile : file, optional, default=None
            A file-like object from which the PDBx/mmCIF file is to be read.
            The file is not closed after reading.
        url : str, optional, default=None
            A URL specifying the internet location from which the file contents should be retrieved.  The format is
            determined automatically by looking for a filename extension in the URL, or if that is ambiguous, by looking
            at the file content.
        pdbid : str, optional, default=None
            A four-letter PDB code specifying the structure to be retrieved from the RCSB.

        Notes
        -----
        Only one of structure, filename, pdbfile, pdbxfile, url, or pdbid may be specified or an exception will be thrown.

        Examples
        --------

        Start from a filename.

        >>> filename = resource_filename('pdbfixer', 'tests/data/test.pdb')
        >>> fixer = PDBFixer(filename=filename)

        Start from a file object.

        >>> with open(filename) as f:
        ...     fixer = PDBFixer(pdbfile=f)

        Start from a URL.

        >>> fixer = PDBFixer(url='http://www.rcsb.org/pdb/files/1VII.pdb')

        Start from a PDB code.

        >>> fixer = PDBFixer(pdbid='1VII')

        """

        # Check to make sure only one option has been specified.
        if bool(filename) + bool(pdbfile) + bool(pdbxfile) + bool(url) + bool(pdbid) != 1:
            raise Exception("Exactly one option [filename, pdbfile, pdbxfile, url, pdbid] must be specified.")

        self.source = None
        if pdbid:
            # A PDB id has been specified.
            url = 'http://www.rcsb.org/pdb/files/%s.pdb' % pdbid
        if filename:
            # A local file has been specified.
            self.source = filename
            file = open(filename, 'r')
            if _guessFileFormat(file, filename) == 'pdbx':
                self._initializeFromPDBx(file)
            else:
                self._initializeFromPDB(file)
            file.close()
        elif pdbfile:
            # A file-like object has been specified.
            self._initializeFromPDB(pdbfile)
        elif pdbxfile:
            # A file-like object has been specified.
            self._initializeFromPDBx(pdbxfile)
        elif url:
            # A URL has been specified.
            self.source = url
            file = urlopen(url)
            contents = file.read().decode('utf-8')
            file.close()
            file = StringIO(contents)
            if _guessFileFormat(file, url) == 'pdbx':
                self._initializeFromPDBx(contents)
            else:
                self._initializeFromPDB(StringIO(contents))

        # Check the structure has some atoms in it.
        atoms = list(self.topology.atoms())
        if len(atoms) == 0:
            raise Exception("Structure contains no atoms.")

        # Load the templates.

        self.templates = {}
        templatesPath = os.path.join(os.path.dirname(__file__), 'templates')
        for file in os.listdir(templatesPath):
            templatePdb = app.PDBFile(os.path.join(templatesPath, file))
            name = next(templatePdb.topology.residues()).name
            self.templates[name] = templatePdb

    def _initializeFromPDB(self, file):
        """Initialize this object by reading a PDB file."""

        structure = PdbStructure(file)
        pdb = app.PDBFile(structure)
        self.topology = pdb.topology
        self.positions = pdb.positions
        self.sequences = [Sequence(s.chain_id, s.residues) for s in structure.sequences]
        self.modifiedResidues = [ModifiedResidue(r.chain_id, r.number, r.residue_name, r.standard_name) for r in structure.modified_residues]

    def _initializeFromPDBx(self, file):
        """Initialize this object by reading a PDBx/mmCIF file."""

        pdbx = app.PDBxFile(file)
        self.topology = pdbx.topology
        self.positions = pdbx.positions

        # PDBxFile doesn't record the information about sequence or modified residues, so we need to read them separately.

        file.seek(0)
        reader = PdbxReader(file)
        data = []
        reader.read(data)
        block = data[0]

        # Load the sequence data.

        sequenceData = block.getObj('entity_poly_seq')
        sequences = {}
        if sequenceData is not None:
            entityIdCol = sequenceData.getAttributeIndex('entity_id')
            residueCol = sequenceData.getAttributeIndex('mon_id')
            for row in sequenceData.getRowList():
                entityId = row[entityIdCol]
                residue = row[residueCol]
                if entityId not in sequences:
                    sequences[entityId] = []
                sequences[entityId].append(residue)

        # Sequences are stored by "entity".  There could be multiple chains that are all the same entity, so we need to
        # convert from entities to chains.

        asymData = block.getObj('struct_asym')
        self.sequences = []
        if asymData is not None:
            asymIdCol = asymData.getAttributeIndex('id')
            entityIdCol = asymData.getAttributeIndex('entity_id')
            for row in asymData.getRowList():
                asymId = row[asymIdCol]
                entityId = row[entityIdCol]
                if entityId in sequences:
                    self.sequences.append(Sequence(asymId, sequences[entityId]))

        # Load the modified residues.

        modData = block.getObj('pdbx_struct_mod_residue')
        self.modifiedResidues = []
        if modData is not None:
            asymIdCol = modData.getAttributeIndex('label_asym_id')
            resNameCol = modData.getAttributeIndex('label_comp_id')
            resNumCol = modData.getAttributeIndex('auth_seq_id')
            standardResCol = modData.getAttributeIndex('parent_comp_id')
            if -1 not in (asymIdCol, resNameCol, resNumCol, standardResCol):
                for row in modData.getRowList():
                    self.modifiedResidues.append(ModifiedResidue(row[asymIdCol], int(row[resNumCol]), row[resNameCol], row[standardResCol]))

    def _addAtomsToTopology(self, heavyAtomsOnly, omitUnknownMolecules):
        """Create a new Topology in which missing atoms have been added.

        Parameters
        ----------
        heavyAtomsOnly : bool
            If True, only heavy atoms will be added to the topology.
        omitUnknownMolecules : bool
            If True, unknown molecules will be omitted from the topology.

        Returns
        -------
        newTopology : openmm.app.Topology
            A new Topology object containing atoms from the old.
        newPositions : list of openmm.unit.Quantity with units compatible with nanometers
            Atom positions for the new Topology object.
        newAtoms : openmm.app.Topology.Atom
            New atom objects.
        existingAtomMap : dict
            Mapping from old atoms to new atoms.

        """

        newTopology = app.Topology()
        newPositions = []*unit.nanometer
        newAtoms = []
        existingAtomMap = {}
        addedAtomMap = {}
        addedOXT = []
        residueCenters = [self._computeResidueCenter(res).value_in_unit(unit.nanometers) for res in self.topology.residues()]*unit.nanometers
        for chain in self.topology.chains():
            if omitUnknownMolecules and not any(residue.name in self.templates for residue in chain.residues()):
                continue
            chainResidues = list(chain.residues())
            newChain = newTopology.addChain(chain.id)
            for indexInChain, residue in enumerate(chain.residues()):

                # Insert missing residues here.

                if (chain.index, indexInChain) in self.missingResidues:
                    insertHere = self.missingResidues[(chain.index, indexInChain)]
                    endPosition = self._computeResidueCenter(residue)
                    if indexInChain > 0:
                        startPosition = self._computeResidueCenter(chainResidues[indexInChain-1])
                        loopDirection = _findUnoccupiedDirection((startPosition+endPosition)/2, residueCenters)
                    else:
                        outward = _findUnoccupiedDirection(endPosition, residueCenters)*unit.nanometers
                        norm = unit.norm(outward)
                        if norm > 0*unit.nanometer:
                            outward *= len(insertHere)*0.5*unit.nanometer/norm
                        startPosition = endPosition+outward
                        loopDirection = None
                    firstIndex = int(residue.id)-len(insertHere)
                    self._addMissingResiduesToChain(newChain, insertHere, startPosition, endPosition, loopDirection, residue, newAtoms, newPositions, firstIndex)

                # Create the new residue and add existing heavy atoms.

                newResidue = newTopology.addResidue(residue.name, newChain, residue.id, residue.insertionCode)
                for atom in residue.atoms():
                    if not heavyAtomsOnly or (atom.element is not None and atom.element != hydrogen):
                        if atom.name == 'OXT' and (chain.index, indexInChain+1) in self.missingResidues:
                            continue # Remove terminal oxygen, since we'll add more residues after this one
                        newAtom = newTopology.addAtom(atom.name, atom.element, newResidue)
                        existingAtomMap[atom] = newAtom
                        newPositions.append(self.positions[atom.index])
                if residue in self.missingAtoms:

                    # Find corresponding atoms in the residue and the template.

                    template = self.templates[residue.name]
                    atomPositions = dict((atom.name, self.positions[atom.index]) for atom in residue.atoms())
                    points1 = []
                    points2 = []
                    for atom in template.topology.atoms():
                        if atom.name in atomPositions:
                            points1.append(atomPositions[atom.name].value_in_unit(unit.nanometer))
                            points2.append(template.positions[atom.index].value_in_unit(unit.nanometer))

                    # Compute the optimal transform to overlay them.

                    (translate2, rotate, translate1) = _overlayPoints(points1, points2)

                    # Add the missing atoms.

                    addedAtomMap[residue] = {}
                    for atom in self.missingAtoms[residue]:
                        newAtom = newTopology.addAtom(atom.name, atom.element, newResidue)
                        newAtoms.append(newAtom)
                        addedAtomMap[residue][atom] = newAtom
                        templatePosition = template.positions[atom.index].value_in_unit(unit.nanometer)
                        newPositions.append((mm.Vec3(*np.dot(rotate, templatePosition+translate2))+translate1)*unit.nanometer)
                if residue in self.missingTerminals:
                    terminalsToAdd = self.missingTerminals[residue]
                else:
                    terminalsToAdd = None

                # If this is the end of the chain, add any missing residues that come after it.

                if residue == chainResidues[-1] and (chain.index, indexInChain+1) in self.missingResidues:
                    insertHere = self.missingResidues[(chain.index, indexInChain+1)]
                    if len(insertHere) > 0:
                        startPosition = self._computeResidueCenter(residue)
                        outward = _findUnoccupiedDirection(startPosition, residueCenters)*unit.nanometers
                        norm = unit.norm(outward)
                        if norm > 0*unit.nanometer:
                            outward *= len(insertHere)*0.5*unit.nanometer/norm
                        endPosition = startPosition+outward
                        firstIndex = int(residue.id)+1
                        self._addMissingResiduesToChain(newChain, insertHere, startPosition, endPosition, None, residue, newAtoms, newPositions, firstIndex)
                        newResidue = list(newChain.residues())[-1]
                        if newResidue.name in proteinResidues:
                            terminalsToAdd = ['OXT']
                        else:
                            terminalsToAdd = None

                # If a terminal OXT is missing, add it.

                if terminalsToAdd is not None:
                    atomPositions = dict((atom.name, newPositions[atom.index].value_in_unit(unit.nanometer)) for atom in newResidue.atoms())
                    if 'OXT' in terminalsToAdd:
                        newAtom = newTopology.addAtom('OXT', oxygen, newResidue)
                        newAtoms.append(newAtom)
                        addedOXT.append(newAtom)
                        d_ca_o = atomPositions['O']-atomPositions['CA']
                        d_ca_c = atomPositions['C']-atomPositions['CA']
                        d_ca_c /= unit.sqrt(unit.dot(d_ca_c, d_ca_c))
                        v = d_ca_o - d_ca_c*unit.dot(d_ca_c, d_ca_o)
                        newPositions.append((atomPositions['O']+2*v)*unit.nanometer)
        newTopology.setUnitCellDimensions(self.topology.getUnitCellDimensions())
        newTopology.createStandardBonds()
        newTopology.createDisulfideBonds(newPositions)

        # Add the bonds between atoms in heterogens.

        for a1,a2 in self.topology.bonds():
            if a1 in existingAtomMap and a2 in existingAtomMap and (a1.residue.name not in app.Topology._standardBonds or a2.residue.name not in app.Topology._standardBonds):
                newTopology.addBond(existingAtomMap[a1], existingAtomMap[a2])

        # Return the results.

        return (newTopology, newPositions, newAtoms, existingAtomMap)

    def _computeResidueCenter(self, residue):
        """Compute the centroid of a residue."""
        return unit.sum([self.positions[atom.index] for atom in residue.atoms()])/len(list(residue.atoms()))

    def _addMissingResiduesToChain(self, chain, residueNames, startPosition, endPosition, loopDirection, orientTo, newAtoms, newPositions, firstIndex):
        """Add a series of residues to a chain."""
        orientToPositions = dict((atom.name, self.positions[atom.index]) for atom in orientTo.atoms())
        if loopDirection is None:
            loopDirection = mm.Vec3(0, 0, 0)

        # We'll add the residues in an arc connecting the endpoints.  Figure out the height of that arc.

        length = unit.norm(endPosition-startPosition)
        numResidues = len(residueNames)
        if length > numResidues*0.3*unit.nanometers:
            loopHeight = 0*unit.nanometers
        else:
            loopHeight = (numResidues*0.3*unit.nanometers-length)/2

        # Add the residues.

        for i, residueName in enumerate(residueNames):
            template = self.templates[residueName]

            # Find a translation that best matches the adjacent residue.

            points1 = []
            points2 = []
            for atom in template.topology.atoms():
                if atom.name in orientToPositions:
                    points1.append(orientToPositions[atom.name].value_in_unit(unit.nanometer))
                    points2.append(template.positions[atom.index].value_in_unit(unit.nanometer))
            (translate2, rotate, translate1) = _overlayPoints(points1, points2)

            # Create the new residue.

            newResidue = chain.topology.addResidue(residueName, chain, "%d" % ((firstIndex+i)%10000))
            fraction = (i+1.0)/(numResidues+1.0)
            translate = startPosition + (endPosition-startPosition)*fraction + loopHeight*math.sin(fraction*math.pi)*loopDirection
            templateAtoms = list(template.topology.atoms())
            if newResidue == next(chain.residues()):
                templateAtoms = [atom for atom in templateAtoms if atom.name not in ('P', 'OP1', 'OP2')]
            for atom in templateAtoms:
                newAtom = chain.topology.addAtom(atom.name, atom.element, newResidue)
                newAtoms.append(newAtom)
                templatePosition = template.positions[atom.index].value_in_unit(unit.nanometer)
                newPositions.append(mm.Vec3(*np.dot(rotate, templatePosition))*unit.nanometer+translate)

    def removeChains(self, chainIndices=None, chainIds=None):
        """Remove a set of chains from the structure.

        Parameters
        ----------
        chainIndices : list of int, optional, default=None
            List of indices of chains to remove.
        chainIds : list of str, optional, default=None
            List of chain ids of chains to remove.

        Examples
        --------

        Load a PDB file with two chains and eliminate the second chain.

        >>> fixer = PDBFixer(pdbid='4J7F')
        >>> fixer.removeChains(chainIndices=[1])

        Load a PDB file with two chains and eliminate chains named 'B' and 'D'.

        >>> fixer = PDBFixer(pdbid='4J7F')
        >>> fixer.removeChains(chainIds=['B','D'])

        """
        modeller = app.Modeller(self.topology, self.positions)
        allChains = list(self.topology.chains())

        if chainIndices == None:
            chainIndices = list()
        if chainIds != None:
            # Add all chains that match the selection to the list.
            for (chainNumber, chain) in enumerate(allChains):
                if chain.id in chainIds:
                    chainIndices.append(chainNumber)
            # Ensure only unique entries remain.
            chainIndices = list(set(chainIndices))

        # Do nothing if no chains will be deleted.
        if len(chainIndices) == 0:
            return

        modeller.delete(allChains[i] for i in chainIndices)
        self.topology = modeller.topology
        self.positions = modeller.positions

        return

    def findMissingResidues(self):
        """Find residues that are missing from the structure.

        The results are stored into the missingResidues field, which is a dict.  Each key is a tuple consisting of
        the index of a chain, and the residue index within that chain at which new residues should be inserted.
        The corresponding value is a list of the names of residues to insert there.

        Examples
        --------

        >>> fixer = PDBFixer(pdbid='1VII')
        >>> fixer.findMissingResidues()
        >>> missing_residues = fixer.missingResidues

        """
        chains = [c for c in self.topology.chains() if len(list(c.residues())) > 0]
        chainWithGaps = {}

        # Find the sequence of each chain, with gaps for missing residues.

        for chain in chains:
            residues = list(chain.residues())
            ids = [int(r.id) for r in residues]
            for i, res in enumerate(residues):
                if res.insertionCode not in ('', ' '):
                    for j in range(i, len(residues)):
                        ids[j] += 1
            minResidue = min(ids)
            maxResidue = max(ids)
            chainWithGaps[chain] = [None]*(maxResidue-minResidue+1)
            for r, id in zip(residues, ids):
                chainWithGaps[chain][id-minResidue] = r.name

        # Try to find the chain that matches each sequence.

        chainSequence = {}
        chainOffset = {}
        for sequence in self.sequences:
            for chain in chains:
                if chain.id != sequence.chainId:
                    continue
                if chain in chainSequence:
                    continue
                for offset in range(len(sequence.residues)-len(chainWithGaps[chain])+1):
                    if all(a == b or b == None for a,b in zip(sequence.residues[offset:], chainWithGaps[chain])):
                        chainSequence[chain] = sequence
                        chainOffset[chain] = offset
                        break
                if chain in chainSequence:
                    break

        # Now build the list of residues to add.

        self.missingResidues = {}
        for chain in self.topology.chains():
            if chain in chainSequence:
                offset = chainOffset[chain]
                sequence = chainSequence[chain].residues
                gappedSequence = chainWithGaps[chain]
                index = 0
                for i in range(len(sequence)):
                    if i < offset or i >= len(gappedSequence)+offset or gappedSequence[i-offset] is None:
                        key = (chain.index, index)
                        if key not in self.missingResidues:
                            self.missingResidues[key] = []
                        residueName = sequence[i]
                        if residueName in substitutions:
                            residueName = substitutions[sequence[i]]
                        self.missingResidues[key].append(residueName)
                    else:
                        index += 1

    def findNonstandardResidues(self):
        """Identify non-standard residues found in the structure, and select standard residues to replace them with.

        The results are stored into the nonstandardResidues field, which is a map of Residue objects to the names
        of suggested replacement residues.

        Examples
        --------

        Find nonstandard residues.

        >>> fixer = PDBFixer(pdbid='1YRI')
        >>> fixer.findNonstandardResidues()
        >>> nonstandard_residues = fixer.nonstandardResidues

        """

        # First find residues based on our table of standard substitutions.

        nonstandard = dict((r, substitutions[r.name]) for r in self.topology.residues() if r.name in substitutions)

        # Now add ones based on MODRES records.

        modres = dict(((m.chainId, str(m.number), m.residueName), m.standardName) for m in self.modifiedResidues)
        for chain in self.topology.chains():
            for residue in chain.residues():
                key = (chain.id, residue.id, residue.name)
                if key in modres:
                    replacement = modres[key]
                    if replacement == 'DU':
                        replacement = 'DT'
                    if replacement in self.templates:
                        nonstandard[residue] = replacement
        self.nonstandardResidues = [(r, nonstandard[r]) for r in sorted(nonstandard, key=lambda r: r.index)]

    def replaceNonstandardResidues(self):
        """Replace every residue listed in the nonstandardResidues field with the specified standard residue.

        Notes
        -----
        You must have first called findNonstandardResidues() to identify nonstandard residues.

        Examples
        --------

        Find and replace nonstandard residues using replacement templates stored in the 'templates' field of PDBFixer object.

        >>> fixer = PDBFixer(pdbid='1YRI')
        >>> fixer.findNonstandardResidues()
        >>> fixer.replaceNonstandardResidues()

        """
        if len(self.nonstandardResidues) > 0:
            deleteAtoms = []

            # Find atoms that should be deleted.

            for residue, replaceWith in self.nonstandardResidues:
                residue.name = replaceWith
                template = self.templates[replaceWith]
                standardAtoms = set(atom.name for atom in template.topology.atoms())
                for atom in residue.atoms():
                    if atom.element in (None, hydrogen) or atom.name not in standardAtoms:
                        deleteAtoms.append(atom)

            # Delete them.

            modeller = app.Modeller(self.topology, self.positions)
            modeller.delete(deleteAtoms)
            self.topology = modeller.topology
            self.positions = modeller.positions


    def applyMutations(self, mutations, chain_id):
        """Apply a list of amino acid substitutions to make a mutant protein.

        Parameters
        ----------
        mutations : list of strings
            Each string must include the resName (original), index,
            and resName (target).  For example, ALA-133-GLY will mutate
            alanine 133 to glycine.
        chain_id : str
            String based chain ID of the single chain you wish to mutate.

        Notes
        -----

        We require three letter codes to avoid possible ambiguitities.
        We can't guarantee that the resulting model is a good one; for
        significant changes in sequence, you should probably be using
        a standalone homology modelling tool.

        Examples
        --------

        Find nonstandard residues.

        >>> fixer = PDBFixer(pdbid='1VII')
        >>> fixer.applyMutations(["ALA-57-GLY"], "A")
        >>> fixer.findMissingResidues()
        >>> fixer.findMissingAtoms()
        >>> fixer.addMissingAtoms()
        >>> fixer.addMissingHydrogens(7.0)

        """
        # Retrieve all residues that match the specified chain_id.
        # NOTE: Multiple chains may have the same chainid, but must have unique resSeq entries.
        resSeq_to_residue = dict() # resSeq_to_residue[resid] is the residue in the requested chain corresponding to residue identifier 'resid'
        for chain in self.topology.chains():
            if chain.id == chain_id:
                for residue in chain.residues():
                    resSeq_to_residue[int(residue.id)] = residue

        # Make a map of residues to mutate based on requested mutation list.
        residue_map = dict() # residue_map[residue] is the name of the new residue to mutate to, if a mutation is desired
        for mut_str in mutations:
            old_name, resSeq, new_name = mut_str.split("-")
            resSeq = int(resSeq)

            if resSeq not in resSeq_to_residue:
                raise(KeyError("Cannot find chain %s residue %d in system!" % (chain_id, resSeq)))

            residue = resSeq_to_residue[resSeq] # retrieve the requested residue

            if residue.name != old_name:
                raise(ValueError("You asked to mutate chain %s residue %d name %s, but that residue is actually %s!" % (chain_id, resSeq, old_name, residue.name)))

            try:
                template = self.templates[new_name]
            except KeyError:
                raise(KeyError("Cannot find residue %s in template library!" % new_name))

            # Store mutation
            residue_map[residue] = new_name

        # If there are mutations to be made, make them.
        if len(residue_map) > 0:            
            deleteAtoms = [] # list of atoms to delete

            # Find atoms that should be deleted.
            for residue in residue_map.keys():
                replaceWith = residue_map[residue]
                residue.name = replaceWith
                template = self.templates[replaceWith]
                standardAtoms = set(atom.name for atom in template.topology.atoms())
                for atom in residue.atoms():
                    if atom.element in (None, hydrogen) or atom.name not in standardAtoms:
                        deleteAtoms.append(atom)

            # Delete atoms queued to be deleted.
            modeller = app.Modeller(self.topology, self.positions)
            modeller.delete(deleteAtoms)
            self.topology = modeller.topology
            self.positions = modeller.positions


    def findMissingAtoms(self):
        """Find heavy atoms that are missing from the structure.

        The results are stored into two fields: missingAtoms and missingTerminals.  Each of these is a dict whose keys
        are Residue objects and whose values are lists of atom names.  missingAtoms contains standard atoms that should
        be present in any residue of that type.  missingTerminals contains terminal atoms that should be present at the
        start or end of a chain.

        Notes
        -----
        You must have first called findMissingResidues().

        Examples
        --------

        Find missing heavy atoms in Abl kinase structure.

        >>> fixer = PDBFixer(pdbid='2F4J')
        >>> fixer.findMissingResidues()
        >>> fixer.findMissingAtoms()
        >>> # Retrieve missing atoms.
        >>> missingAtoms = fixer.missingAtoms
        >>> # Retrieve missing terminal atoms.
        >>> missingTerminals = fixer.missingTerminals

        """
        missingAtoms = {}
        missingTerminals = {}

        # Loop over residues.

        for chain in self.topology.chains():
            chainResidues = list(chain.residues())
            for residue in chain.residues():
                if residue.name in self.templates:
                    template = self.templates[residue.name]
                    atomNames = set(atom.name for atom in residue.atoms())
                    templateAtoms = list(template.topology.atoms())
                    if residue == chainResidues[0] and (chain.index, 0) not in self.missingResidues:
                        templateAtoms = [atom for atom in templateAtoms if atom.name not in ('P', 'OP1', 'OP2')]

                    # Add atoms from the template that are missing.

                    missing = []
                    for atom in templateAtoms:
                        if atom.name not in atomNames:
                            missing.append(atom)
                    if len(missing) > 0:
                        missingAtoms[residue] = missing

                    # Add missing terminal atoms.

                    terminals = []
                    if residue == chainResidues[-1] and (chain.index, len(chainResidues)) not in self.missingResidues:
                        templateNames = set(atom.name for atom in template.topology.atoms())
                        if 'OXT' not in atomNames and all(name in templateNames for name in ['C', 'O', 'CA']):
                            terminals.append('OXT')
                        if len(terminals) > 0:
                            missingTerminals[residue] = terminals
        self.missingAtoms = missingAtoms
        self.missingTerminals = missingTerminals

    def addMissingAtoms(self, seed=None):
        """Add all missing heavy atoms, as specified by the missingAtoms, missingTerminals, and missingResidues fields.

        Parameters
        ----------
        seed : int
            Integer to set the random seed number of the integrator used in the minimization of the
            coordinates of the newly-added atoms.

        Notes
        -----
        You must already have called findMissingAtoms() to have identified atoms to be added.

        Examples
        --------

        Find missing heavy atoms in Abl kinase structure.

        >>> fixer = PDBFixer(pdbid='2F4J')
        >>> fixer.findMissingResidues()
        >>> fixer.findMissingAtoms()
        >>> fixer.addMissingAtoms()

        """

        # Create a Topology that 1) adds missing atoms, 2) removes all hydrogens, and 3) removes unknown molecules.

        (newTopology, newPositions, newAtoms, existingAtomMap) = self._addAtomsToTopology(True, True)
        if len(newAtoms) == 0:

            # No atoms were added, but new bonds might have been created.

            newBonds = set(newTopology.bonds())
            for atom1, atom2 in self.topology.bonds():
                if atom1 in existingAtomMap and atom2 in existingAtomMap:
                    a1 = existingAtomMap[atom1]
                    a2 = existingAtomMap[atom2]
                    if (a1, a2) in newBonds:
                        newBonds.remove((a1, a2))
                    elif (a2, a1) in newBonds:
                        newBonds.remove((a2, a1))

            # Add the new bonds to the original Topology.

            inverseAtomMap = dict((y,x) for (x,y) in existingAtomMap.items())
            for atom1, atom2 in newBonds:
                self.topology.addBond(inverseAtomMap[atom1], inverseAtomMap[atom2])
        else:

            # Create a System for energy minimizing it.

            forcefield = self._createForceField(newTopology, False)
            system = forcefield.createSystem(newTopology)

            # Set any previously existing atoms to be massless, they so won't move.

            for atom in existingAtomMap.values():
                system.setParticleMass(atom.index, 0.0)

            # If any heavy atoms were omitted, add them back to avoid steric clashes.

            nonbonded = [f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce)][0]
            for atom in self.topology.atoms():
                if atom.element not in (None, hydrogen) and atom not in existingAtomMap:
                    system.addParticle(0.0)
                    nonbonded.addParticle([])
                    newPositions.append(self.positions[atom.index])

            # For efficiency, only compute interactions that involve a new atom.

            nonbonded.addInteractionGroup([atom.index for atom in newAtoms], range(system.getNumParticles()))

            # Do an energy minimization.

            integrator = mm.LangevinIntegrator(300*unit.kelvin, 10/unit.picosecond, 5*unit.femtosecond)
            if seed is not None:
                integrator.setRandomNumberSeed(seed)
            # omicverse: pin this to the Reference platform. Upstream lets
            # OpenMM pick the fastest one, which is normally a GPU — and GPU
            # reductions are not order-deterministic, so the rebuilt atom
            # positions differed between runs even with `seed` set, making
            # everything downstream irreproducible. This minimisation only
            # touches the handful of newly added atoms, so the slower platform
            # costs little and buys a reproducible starting structure.
            context = mm.Context(system, integrator,
                                 mm.Platform.getPlatformByName('Reference'))
            context.setPositions(newPositions)
            mm.LocalEnergyMinimizer.minimize(context)
            state = context.getState(getPositions=True)
            if newTopology.getNumResidues() > 1:
                nearest = self._findNearestDistance(context, newTopology, newAtoms)
                if nearest < 0.13:

                    # Some atoms are very close together.  Run some dynamics while slowly increasing the strength of the
                    # repulsive interaction to try to improve the result.

                    for i in range(10):
                        context.setParameter('C', 0.15*(i+1))
                        integrator.step(200)
                        d = self._findNearestDistance(context, newTopology, newAtoms)
                        if d > nearest:
                            nearest = d
                            state = context.getState(getPositions=True)
                            if nearest >= 0.13:
                                break
                    context.setState(state)
                    context.setParameter('C', 1.0)
                    mm.LocalEnergyMinimizer.minimize(context)
                    state = context.getState(getPositions=True)

            # Now create a new Topology, including all atoms from the original one and adding the missing atoms.

            (newTopology2, newPositions2, newAtoms2, existingAtomMap2) = self._addAtomsToTopology(False, False)

            # Copy over the minimized positions for the new atoms.

            for a1, a2 in zip(newAtoms, newAtoms2):
                newPositions2[a2.index] = state.getPositions()[a1.index]
            self.topology = newTopology2
            self.positions = newPositions2

    def removeHeterogens(self, keepWater=True):
        """Remove all heterogens from the structure.

        Parameters
        ----------
        keepWater : bool, optional, default=True
            If True, water molecules will not be removed.

        Examples
        --------

        Remove heterogens in Abl structure complexed with imatinib.

        >>> fixer = PDBFixer(pdbid='2F4J')
        >>> fixer.removeHeterogens(keepWater=False)

        """

        keep = set(proteinResidues).union(dnaResidues).union(rnaResidues)
        keep.add('N')
        keep.add('UNK')
        if keepWater:
            keep.add('HOH')
        toDelete = []
        for residue in self.topology.residues():
            if residue.name not in keep:
                toDelete.append(residue)
        modeller = app.Modeller(self.topology, self.positions)
        modeller.delete(toDelete)
        self.topology = modeller.topology
        self.positions = modeller.positions

    def addMissingHydrogens(self, pH=7.0):
        """Add missing hydrogen atoms to the structure.

        Parameters
        ----------
        pH : float, optional, default=7.0
            The pH based on which to select hydrogens.

        Notes
        -----
        No extensive electrostatic analysis is performed; only default residue pKas are used.

        Examples
        --------

        Examples
        --------

        Add missing hydrogens appropriate for pH 8.

        >>> fixer = PDBFixer(pdbid='1VII')
        >>> fixer.addMissingHydrogens(pH=8.0)

        """
        modeller = app.Modeller(self.topology, self.positions)
        modeller.addHydrogens(pH=pH)
        self.topology = modeller.topology
        self.positions = modeller.positions

    def addSolvent(self, boxSize=None, padding=None, boxVectors=None, positiveIon='Na+', negativeIon='Cl-', ionicStrength=0*unit.molar):
        """Add a solvent box surrounding the structure.

        Parameters
        ----------
        boxSize : openmm.Vec3, optional, default=None
            The size of the box to fill with water.  If specified, padding and boxVectors must not be specified.
        padding : openmm.unit.Quantity compatible with nanometers, optional, default=None
            Padding around macromolecule for filling box with water.  If specified, boxSize and boxVectors must not be specified.
        boxVectors : 3-tuple of openmm.Vec3, optional, default=None
            Three vectors specifying the geometry of the box. If specified, padding and boxSize must not be specified.
        positiveIon : str, optional, default='Na+'
            The type of positive ion to add.  Allowed values are 'Cs+', 'K+', 'Li+', 'Na+', and 'Rb+'.
        negativeIon : str, optional, default='Cl-'
            The type of negative ion to add.  Allowed values are 'Cl-', 'Br-', 'F-', and 'I-'.
        ionicStrength : openmm.unit.Quantity with units compatible with molar, optional, default=0*molar
            The total concentration of ions (both positive and negative) to add.  This does not include ions that are added to neutralize the system.

        Examples
        --------

        Add missing residues, heavy atoms, and hydrogens, and then solvate with 10 A padding.

        >>> fixer = PDBFixer(pdbid='1VII')
        >>> fixer.findMissingResidues()
        >>> fixer.findMissingAtoms()
        >>> fixer.addMissingAtoms()
        >>> fixer.addMissingHydrogens(pH=8.0)
        >>> fixer.addSolvent(padding=10*unit.angstrom, ionicStrength=0.050*unit.molar)

        """

        modeller = app.Modeller(self.topology, self.positions)
        forcefield = self._createForceField(self.topology, True)
        modeller.addSolvent(forcefield, padding=padding, boxSize=boxSize, boxVectors=boxVectors, positiveIon=positiveIon, negativeIon=negativeIon, ionicStrength=ionicStrength)
        chains = list(modeller.topology.chains())
        if len(chains) == 1:
            chains[0].id = 'A'
        else:
            chains[-1].id = chr(ord(chains[-2].id)+1)
        self.topology = modeller.topology
        self.positions = modeller.positions

    def addMembrane(self, lipidType='POPC', membraneCenterZ=0*unit.nanometer, minimumPadding=1*unit.nanometer, positiveIon='Na+', negativeIon='Cl-', ionicStrength=0*unit.molar):
        """Add a lipid membrane to the structure.

        This method adds both lipids and water, so you should call either addSolvent() or addMembrane(),
        but not both.  See Modeller.addMembrane() for more details.

        Parameters
        ----------
        lipidType : string='POPC'
            the type of lipid to use.  Supported values are 'POPC', 'POPE', 'DLPC', 'DLPE', 'DMPC', 'DOPC', and 'DPPC'.
        membraneCenterZ: distance=0*nanometer
            the position along the Z axis of the center of the membrane
        minimumPadding : distance=1*nanometer
            the padding distance to use
        positiveIon : str, optional, default='Na+'
            The type of positive ion to add.  Allowed values are 'Cs+', 'K+', 'Li+', 'Na+', and 'Rb+'.
        negativeIon : str, optional, default='Cl-'
            The type of negative ion to add.  Allowed values are 'Cl-', 'Br-', 'F-', and 'I-'.
        ionicStrength : openmm.unit.Quantity with units compatible with molar, optional, default=0*molar
            The total concentration of ions (both positive and negative) to add.  This does not include ions that are added to neutralize the system.
        """
        modeller = app.Modeller(self.topology, self.positions)
        forcefield = self._createForceField(self.topology, True)
        modeller.addMembrane(forcefield, lipidType=lipidType, minimumPadding=minimumPadding, positiveIon=positiveIon, negativeIon=negativeIon, ionicStrength=ionicStrength)
        chains = list(modeller.topology.chains())
        if len(chains) == 1:
            chains[0].id = 'A'
        else:
            chains[-1].id = chr(ord(chains[-2].id)+1)
        self.topology = modeller.topology
        self.positions = modeller.positions

    def _createForceField(self, newTopology, water):
        """Create a force field to use for optimizing the positions of newly added atoms."""

        if water:
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
            nonbonded = [f for f in forcefield._forces if isinstance(f, NonbondedGenerator)][0]
            radii = {'H':0.198, 'Li':0.203, 'C':0.340, 'N':0.325, 'O':0.299, 'F':0.312, 'Na':0.333, 'Mg':0.141,
                     'P':0.374, 'S':0.356, 'Cl':0.347, 'K':0.474, 'Br':0.396, 'Rb':0.527, 'I':0.419, 'Cs':0.605}
        else:
            forcefield = app.ForceField(os.path.join(os.path.dirname(__file__), 'soft.xml'))

        # The Topology may contain residues for which the ForceField does not have a template.
        # If so, we need to create new templates for them.

        atomTypes = {}
        bondedToAtom = []
        for atom in newTopology.atoms():
            bondedToAtom.append(set())
        for atom1, atom2 in newTopology.bonds():
            bondedToAtom[atom1.index].add(atom2.index)
            bondedToAtom[atom2.index].add(atom1.index)
        for residue in newTopology.residues():

            # Make sure the ForceField has a template for this residue.

            signature = app.forcefield._createResidueSignature([atom.element for atom in residue.atoms()])
            if signature in forcefield._templateSignatures:
                if any(matchResidue(residue, t, bondedToAtom) is not None for t in forcefield._templateSignatures[signature]):
                    continue

            # Create a new template.

            resName = "extra_"+residue.name
            template = app.ForceField._TemplateData(resName)
            forcefield._templates[resName] = template
            indexInResidue = {}
            for atom in residue.atoms():
                element = atom.element
                typeName = 'extra_'+element.symbol
                if element not in atomTypes:
                    atomTypes[element] = app.ForceField._AtomType(typeName, '', 0.0, element)
                    forcefield._atomTypes[typeName] = atomTypes[element]
                    if water:
                        # Select a reasonable vdW radius for this atom type.

                        if element.symbol in radii:
                            sigma = radii[element.symbol]
                        else:
                            sigma = 0.5
                        nonbonded.registerAtom({'type':typeName, 'charge':'0', 'sigma':str(sigma), 'epsilon':'0'})
                indexInResidue[atom.index] = len(template.atoms)
                template.atoms.append(app.ForceField._TemplateAtomData(atom.name, typeName, element))
            for atom in residue.atoms():
                for bondedTo in bondedToAtom[atom.index]:
                    if bondedTo in indexInResidue:
                        b = (indexInResidue[atom.index], indexInResidue[bondedTo])
                        if b[0] < b[1]:
                            template.bonds.append(b)
                            template.atoms[b[0]].bondedTo.append(b[1])
                            template.atoms[b[1]].bondedTo.append(b[0])
                    else:
                        b = indexInResidue[atom.index]
                        template.externalBonds.append(b)
                        template.atoms[b].externalBonds += 1
            if signature in forcefield._templateSignatures:
                forcefield._templateSignatures[signature].append(template)
            else:
                forcefield._templateSignatures[signature] = [template]
        return forcefield

    def _findNearestDistance(self, context, topology, newAtoms):
        """Given a set of newly added atoms, find the closest distance between one of those atoms and another atom."""

        positions = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        atomResidue = [atom.residue for atom in topology.atoms()]
        nearest = sys.float_info.max
        for atom in newAtoms:
            p = positions-positions[atom.index]
            dist = math.sqrt(min(np.dot(p[i], p[i]) for i in range(len(atomResidue)) if atomResidue[i] != atom.residue))
            if dist < nearest:
                nearest = dist
        return nearest
