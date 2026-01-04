from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from anastruct import SystemElements
import re

from visualization import generate_frame_diagrams

app = FastAPI(title="Frame Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Data Models ----------


class BeamInput(BaseModel):
    span: float
    load: float  # kN/m


class Support(BaseModel):
    nodeId: str
    type: str  # fixed, pin, roller, hinge

    class Config:
        extra = "ignore"


class Load(BaseModel):
    id: str
    type: str  # point, distributed, moment
    nodeId: Optional[str] = None
    elementId: Optional[str] = None
    magnitude: float
    unit: Optional[str] = None   # allow missing unit
    direction: str  # x, y

    class Config:
        extra = "ignore"


class Element(BaseModel):
    i: int
    j: int
    E: Optional[float] = None
    A: Optional[float] = None
    I: Optional[float] = None
    EA: Optional[float] = None
    EI: Optional[float] = None


class FrameInput(BaseModel):
    nodes: List[List[float]]
    elements: List[Element]
    supports: List[Support]
    loads: List[Load]


# ---------- Helper: Coordinate Transform ----------


def sort_frame_data(data: FrameInput) -> FrameInput:
    """
    Only flips Y-axis for pixel coordinates. NO node reordering.
    Preserves frontend node indices so supports/loads still refer to the same node index.
    """
    y_values = [float(coords[1]) for coords in data.nodes]
    x_values = [float(coords[0]) for coords in data.nodes]
    y_min, y_max = min(y_values), max(y_values)
    x_min, x_max = min(x_values), max(x_values)

    coord_range = max(y_max - y_min, x_max - x_min)
    needs_flip = coord_range > 50

    new_nodes_list = []
    for coords in data.nodes:
        cx = float(coords[0])
        cy = float(coords[1])
        cy_new = (y_max - cy + y_min) if needs_flip else cy
        new_nodes_list.append([cx, cy_new])

    print(f"\n{'='*60}")
    print("COORDINATE TRANSFORM DEBUG")
    print(f"{'='*60}")
    print(f"Y-flip applied: {needs_flip}")
    print(f"Coordinate range: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
    print(f"Original nodes: {data.nodes}")
    print(f"Transformed nodes: {new_nodes_list}")
    print(f"Supports: {[(s.nodeId, s.type) for s in data.supports]}")
    print(f"Loads: {[(ld.nodeId or ld.elementId, ld.type, ld.direction) for ld in data.loads]}")
    print(f"{'='*60}\n")

    return FrameInput(
        nodes=new_nodes_list,
        elements=data.elements,
        supports=data.supports,
        loads=data.loads,
    )


# ---------- Helper: Robust ID parsing + mapping ----------


def _num_id(s: str) -> int:
    m = re.search(r"\d+", s)
    if not m:
        raise ValueError(f"Invalid id format: {s}")
    return int(m.group(0))


def node_index_from_nodeId(nodeId: str, n_nodes: int) -> int:
    """Accepts N0..N(n-1) or N1..Nn; returns 0-based index."""
    k = _num_id(nodeId)
    if 1 <= k <= n_nodes:
        return k - 1
    if 0 <= k < n_nodes:
        return k
    raise ValueError(f"nodeId {nodeId} out of range for {n_nodes} nodes")


def elem_index_from_elementId(elementId: str, n_elems: int) -> int:
    """Accepts E0..E(n-1) or E1..En; returns 0-based index."""
    k = _num_id(elementId)
    if 1 <= k <= n_elems:
        return k - 1
    if 0 <= k < n_elems:
        return k
    raise ValueError(f"elementId {elementId} out of range for {n_elems} elements")


def build_id_maps(ss: SystemElements, data: FrameInput, required_node_indices=None):
    """Map a subset of frontend node indices -> anaStruct node_id (by coordinate lookup)."""
    if required_node_indices is None:
        required_node_indices = range(len(data.nodes))

    node_id_map = {}
    missing = []
    for idx in required_node_indices:
        coords = data.nodes[idx]
        nid = ss.find_node_id(coords)
        if nid is None:
            missing.append((idx, coords))
        else:
            node_id_map[idx] = nid

    # If a required node is missing, the model is inconsistent (element endpoint not found).
    if missing:
        missing_str = ", ".join([f"idx={i} coords={c}" for i, c in missing])
        raise ValueError(
            "Some required nodes (element endpoints) were not created in the FEM model. "
            f"This usually indicates duplicate/invalid element endpoints. Missing: {missing_str}"
        )

    return node_id_map


def _dir_to_fx_fy(direction: str, magnitude: float):
    """
    Convert common direction strings into Fx/Fy (kN) for anaStruct.
    
    Uses Cartesian coordinate system (Y-up):
    - "down" = gravity = negative Y force
    - "up" = positive Y force
    - "left" = negative X force
    - "right" = positive X force
    
    This ensures consistency with frontend visualizations where:
    - Editor tab shows load direction labels
    - CAD visualization shows arrows pointing in force direction
    - Matplotlib diagrams show loads in correct direction
    """
    d = (direction or "").lower().strip()
    mag = float(magnitude)

    # Legacy backend behavior: 'y' means downward gravity.
    if d == "y":
        return 0.0, -abs(mag)
    if d == "x":
        return mag, 0.0

    if "down" in d:
        return 0.0, -abs(mag)
    if "up" in d:
        return 0.0, abs(mag)
    if "left" in d:
        return -abs(mag), 0.0
    if "right" in d:
        return abs(mag), 0.0

    # Unknown direction: no force
    return 0.0, 0.0


def _dir_to_udl(direction: str, magnitude: float):
    """
    Convert common direction strings into (q, q_dir) for anaStruct q_load.
    
    Uses Cartesian coordinate system (Y-up):
    - "down" = gravity = negative Y force
    - "up" = positive Y force
    - "left" = negative X force
    - "right" = positive X force
    
    This ensures consistency with frontend visualizations where:
    - Editor tab shows load direction labels
    - CAD visualization shows arrows pointing in force direction
    - Matplotlib diagrams show loads in correct direction
    """
    d = (direction or "").lower().strip()
    mag = float(magnitude)

    # Legacy backend behavior: 'y' means downward gravity.
    if d == "y":
        return -abs(mag), "y"
    if d == "x":
        return mag, "x"

    if "down" in d:
        return -abs(mag), "y"
    if "up" in d:
        return abs(mag), "y"
    if "left" in d:
        return -abs(mag), "x"
    if "right" in d:
        return abs(mag), "x"

    raise ValueError(f"Invalid distributed load direction: {direction}")


# ---------- Debug Endpoint ----------


@app.post("/debug_frame")
async def debug_frame(data: FrameInput):
    out_supports = []
    for s in data.supports:
        idx = node_index_from_nodeId(s.nodeId, len(data.nodes))
        out_supports.append({
            "nodeId": s.nodeId,
            "type": s.type,
            "node_index_0_based": idx,
            "coords": data.nodes[idx],
        })

    out_loads = []
    for ld in data.loads:
        out_loads.append({
            "id": ld.id,
            "type": ld.type,
            "target": ld.nodeId or ld.elementId,
            "direction": ld.direction,
            "magnitude": ld.magnitude,
        })

    out_elements = []
    for e in data.elements:
        out_elements.append({
            "i": e.i,
            "j": e.j,
            "from": data.nodes[e.i],
            "to": data.nodes[e.j],
        })

    return {
        "nodes": data.nodes,
        "node_count": len(data.nodes),
        "elements": out_elements,
        "supports": out_supports,
        "loads": out_loads,
    }


# ---------- Endpoints ----------


@app.post("/analyze")
async def analyze_beam(data: BeamInput):
    ss = SystemElements()
    ss.add_element([[0, 0], [data.span, 0]])
    ss.add_support_hinged(1)
    ss.add_support_roll(2, direction='x')
    ss.q_load(q=-data.load, element_id=1)
    ss.solve()

    M_vals = [m for m in ss.get_element_result_range("moment") if m is not None]
    V_vals = [v for v in ss.get_element_result_range("shear") if v is not None]
    w_vals = [w for w in ss.get_node_result_range("uy") if w is not None]

    max_moment = max(abs(m) for m in M_vals) if M_vals else None
    max_shear = max(abs(v) for v in V_vals) if V_vals else None
    max_deflection_mm = max(abs(w) for w in w_vals) * 1000 if w_vals else None

    return {
        "success": True,
        "results": {
            "max_moment_kNm": max_moment,
            "max_shear_kN": max_shear,
            "max_deflection_mm": max_deflection_mm,
        },
    }


@app.post("/analyze_frame")
async def analyze_frame(data: FrameInput):
    try:
        data = sort_frame_data(data)
        ss = SystemElements()

        # --- Add elements and capture anaStruct element IDs ---
        print(f"\nAdding {len(data.elements)} elements...")
        element_id_map = {}
        for idx, e in enumerate(data.elements):
            calc_EA = e.EA if e.EA else (e.E * e.A if e.E and e.A else None)
            calc_EI = e.EI if e.EI else (e.E * e.I if e.E and e.I else None)
            kwargs = {}
            if calc_EA is not None:
                kwargs["EA"] = calc_EA
            if calc_EI is not None:
                kwargs["EI"] = calc_EI

            node_i_coords = data.nodes[e.i]
            node_j_coords = data.nodes[e.j]
            ana_eid = ss.add_element(location=[node_i_coords, node_j_coords], **kwargs)
            element_id_map[idx] = ana_eid
            print(f"  Element {ana_eid}: NodeIdx {e.i} {node_i_coords} -> NodeIdx {e.j} {node_j_coords}")

        # --- Build node_id_map for element endpoints only ---
        required_nodes = set()
        for e in data.elements:
            required_nodes.add(e.i)
            required_nodes.add(e.j)
        node_id_map = build_id_maps(ss, data, sorted(required_nodes))

        # --- Add supports using node_id_map ---
        print(f"\nAdding {len(data.supports)} supports...")
        support_count = 0
        for s in data.supports:
            try:
                node_idx = node_index_from_nodeId(s.nodeId, len(data.nodes))
                if node_idx not in node_id_map:
                    print(f"  WARN: skipping support at isolated frontend node {s.nodeId} (idx {node_idx})")
                    continue
                nid = node_id_map[node_idx]
                coords = data.nodes[node_idx]

                if s.type == "fixed":
                    ss.add_support_fixed(nid)
                    print(f"  Fixed support at anaNode {nid} (frontend idx {node_idx}) {coords}")
                    support_count += 1
                elif s.type in ["pin", "pinned", "hinge"]:
                    ss.add_support_hinged(nid)
                    print(f"  Pinned/Hinge support at anaNode {nid} (frontend idx {node_idx}) {coords}")
                    support_count += 1
                elif s.type == "roller":
                    ss.add_support_roll(nid, direction='x')
                    print(f"  Roller support at anaNode {nid} (frontend idx {node_idx}) {coords}")
                    support_count += 1
                else:
                    print(f"  WARN: Unknown support type '{s.type}' for {s.nodeId}")
            except Exception as e:
                print(f"  ERROR adding support {s.nodeId}: {e}")

        print(f"Total supports applied: {support_count}")

        # --- Add loads using maps ---
        print(f"\nAdding {len(data.loads)} loads...")
        print(f"Load data received from frontend: {[(l.id, l.type, l.nodeId or l.elementId, l.direction, l.magnitude) for l in data.loads]}")
        load_applied_count = 0
        for ld in data.loads:
            if ld.type in ["point", "moment"] and ld.nodeId:
                try:
                    node_idx = node_index_from_nodeId(ld.nodeId, len(data.nodes))
                    if node_idx not in node_id_map:
                        print(f"  WARN: skipping {ld.type} load at isolated frontend node {ld.nodeId} (idx {node_idx})")
                        continue
                    nid = node_id_map[node_idx]
                    coords = data.nodes[node_idx]

                    if ld.type == "point":
                        fx, fy = _dir_to_fx_fy(ld.direction, ld.magnitude)

                        if fx != 0:
                            ss.point_load(Fx=fx, node_id=nid)
                            print(f"  Point load Fx={fx} kN at anaNode {nid} {coords}")
                        if fy != 0:
                            ss.point_load(Fy=fy, node_id=nid)
                            print(f"  Point load Fy={fy} kN at anaNode {nid} {coords}")
                        load_applied_count += 1

                    elif ld.type == "moment":
                        d = (ld.direction or "").lower()
                        tz = float(ld.magnitude)
                        if "clock" in d and "counter" not in d and "anti" not in d:
                            tz = -abs(tz)
                        elif "counter" in d or "anti" in d:
                            tz = abs(tz)
                        ss.moment_load(Tz=tz, node_id=nid)
                        print(f"  Moment load Tz={ld.magnitude} kNm at anaNode {nid} {coords}")
                        load_applied_count += 1

                except Exception as e:
                    print(f"  ERROR adding load {ld.id}: {e}")

            elif ld.type == "distributed" and ld.elementId:
                try:
                    eidx = elem_index_from_elementId(ld.elementId, len(data.elements))
                    eid = element_id_map[eidx]

                    q, q_dir = _dir_to_udl(ld.direction, ld.magnitude)

                    ss.q_load(q=q, element_id=eid, direction=q_dir)
                    print(f"  Distributed load q={q} kN/m dir={q_dir} on anaElement {eid} (frontend idx {eidx})")
                    load_applied_count += 1
                except Exception as e:
                    print(f"  ERROR adding distributed load {ld.id}: {e}")

        print(f"Total loads applied: {load_applied_count}\n")

        ss.solve()

        # --- Extract results ---
        node_results = ss.get_node_results_system(node_id=0)
        node_disp = (
            {r["id"]: r for r in node_results}
            if isinstance(node_results[0], dict)
            else {r[0]: {"id": r[0], "ux": r[4], "uy": r[5]} for r in node_results}
        )

        element_results = ss.get_element_results(element_id=0)
        elements_output = []

        for ana_eid, e_res in enumerate(element_results, start=1):
            frontend_idx = None
            for k, v in element_id_map.items():
                if v == ana_eid:
                    frontend_idx = k
                    break
            if frontend_idx is None:
                continue

            input_el = data.elements[frontend_idx]
            ana_i_node_id = node_id_map[input_el.i]
            ana_j_node_id = node_id_map[input_el.j]

            ux_i = node_disp.get(ana_i_node_id, {}).get("ux", 0)
            uy_i = node_disp.get(ana_i_node_id, {}).get("uy", 0)
            ux_j = node_disp.get(ana_j_node_id, {}).get("ux", 0)
            uy_j = node_disp.get(ana_j_node_id, {}).get("uy", 0)

            # Get detailed values at 0.1m (10cm) intervals along element
            element_length = e_res["length"]
            interval_m = 0.1  # 10cm intervals
            
            # Generate locations at 10cm intervals
            positions_m = []
            N_values = []
            V_values = []
            M_values = []
            ux_values = []  # horizontal displacements
            uy_values = []  # vertical displacements
            
            num_intervals = max(2, int(element_length / interval_m) + 1)
            
            print(f"\nElement {ana_eid} interval extraction:")
            print(f"  Length: {element_length}m, Intervals: {num_intervals}")
            
            # Get the actual element object from anaStruct
            try:
                element = ss.elements[ana_eid]
                print(f"  Found element object, methods: {dir(element)}")
            except Exception as ex:
                print(f"  ERROR getting element object: {ex}")
                element = None
            
            for i in range(num_intervals):
                pos = min(i * interval_m, element_length)
                positions_m.append(pos)
                
                N = None
                V = None
                M = None
                
                # Try to get forces from element methods
                if element:
                    try:
                        N = element.Nx(pos)  # Axial force
                        V = element.Qy(pos)  # Shear force (y-direction)
                        M = element.M(pos)   # Bending moment
                        print(f"    pos {pos:.3f}m: N={N:.3f}, V={V:.3f}, M={M:.3f}")
                    except Exception as ex:
                        print(f"    WARNING at pos {pos}: {ex}")
                        N = V = M = None
                
                # Fallback: linear interpolation between min and max
                if N is None or V is None or M is None:
                    ratio = pos / element_length if element_length > 0 else 0
                    N = e_res["Nmin"] + (e_res["Nmax"] - e_res["Nmin"]) * ratio
                    V = e_res["Qmin"] + (e_res["Qmax"] - e_res["Qmin"]) * ratio
                    M = e_res["Mmin"] + (e_res["Mmax"] - e_res["Mmin"]) * ratio
                    print(f"    pos {pos:.3f}m (FALLBACK): N={N:.3f}, V={V:.3f}, M={M:.3f}")
                
                N_values.append(float(N))
                V_values.append(float(V))
                M_values.append(float(M))
                
                # Interpolate displacements along the element
                ratio = pos / element_length if element_length > 0 else 0
                ux = ux_i + (ux_j - ux_i) * ratio
                uy = uy_i + (uy_j - uy_i) * ratio
                ux_values.append(float(ux))
                uy_values.append(float(uy))

            elements_output.append({
                "element_id": frontend_idx,
                "frontend_element_index": frontend_idx,
                "node_i": input_el.i,
                "node_j": input_el.j,
                "length_m": e_res["length"],
                "ux_i_m": ux_i,
                "uy_i_m": uy_i,
                "ux_j_m": ux_j,
                "uy_j_m": uy_j,
                "Nmin_kN": e_res["Nmin"],
                "Nmax_kN": e_res["Nmax"],
                "Vmin_kN": e_res["Qmin"],
                "Vmax_kN": e_res["Qmax"],
                "Mmin_kNm": e_res["Mmin"],
                "Mmax_kNm": e_res["Mmax"],
                # New: values at 10cm intervals
                "positions_m": positions_m,
                "N_at_positions_kN": N_values,
                "V_at_positions_kN": V_values,
                "M_at_positions_kNm": M_values,
                "ux_at_positions_m": ux_values,
                "uy_at_positions_m": uy_values,
            })

        return {"success": True, "elements": elements_output}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/analyze_frame/visualize")
async def visualize_frame_results(data: FrameInput):
    try:
        data = sort_frame_data(data)
        ss = SystemElements()

        # Add elements and capture IDs
        element_id_map = {}
        for idx, e in enumerate(data.elements):
            calc_EA = e.EA if e.EA else (e.E * e.A if e.E and e.A else None)
            calc_EI = e.EI if e.EI else (e.E * e.I if e.E and e.I else None)
            kwargs = {}
            if calc_EA is not None:
                kwargs["EA"] = calc_EA
            if calc_EI is not None:
                kwargs["EI"] = calc_EI
            ana_eid = ss.add_element(location=[data.nodes[e.i], data.nodes[e.j]], **kwargs)
            element_id_map[idx] = ana_eid

        required_nodes = set()
        for e in data.elements:
            required_nodes.add(e.i)
            required_nodes.add(e.j)
        node_id_map = build_id_maps(ss, data, sorted(required_nodes))

        # Supports
        for s in data.supports:
            try:
                node_idx = node_index_from_nodeId(s.nodeId, len(data.nodes))
                if node_idx not in node_id_map:
                    continue
                nid = node_id_map[node_idx]
                if s.type == "fixed":
                    ss.add_support_fixed(nid)
                elif s.type in ["pin", "pinned", "hinge"]:
                    ss.add_support_hinged(nid)
                elif s.type == "roller":
                    ss.add_support_roll(nid, direction='x')
            except Exception:
                pass

        # Loads
        for ld in data.loads:
            if ld.type in ["point", "moment"] and ld.nodeId:
                try:
                    node_idx = node_index_from_nodeId(ld.nodeId, len(data.nodes))
                    if node_idx not in node_id_map:
                        continue
                    nid = node_id_map[node_idx]
                    if ld.type == "point":
                        fx, fy = _dir_to_fx_fy(ld.direction, ld.magnitude)
                        if fx != 0:
                            ss.point_load(Fx=fx, node_id=nid)
                        if fy != 0:
                            ss.point_load(Fy=fy, node_id=nid)
                    elif ld.type == "moment":
                        d = (ld.direction or "").lower()
                        tz = float(ld.magnitude)
                        if "clock" in d and "counter" not in d and "anti" not in d:
                            tz = -abs(tz)
                        elif "counter" in d or "anti" in d:
                            tz = abs(tz)
                        ss.moment_load(Tz=tz, node_id=nid)
                except Exception:
                    pass

            elif ld.type == "distributed" and ld.elementId:
                try:
                    eidx = elem_index_from_elementId(ld.elementId, len(data.elements))
                    eid = element_id_map[eidx]

                    q, q_dir = _dir_to_udl(ld.direction, ld.magnitude)

                    ss.q_load(q=q, element_id=eid, direction=q_dir)
                except Exception:
                    pass

        ss.solve()
        diagrams = generate_frame_diagrams(ss)

        has_diagrams = any(v is not None for v in diagrams.values())
        return {"success": has_diagrams, "diagrams": diagrams}

    except Exception as e:
        return {"success": False, "error": str(e), "diagrams": {}}


@app.get("/")
async def root():
    return {"status": "Backend running", "ready": True}


@app.post("/debug_element_methods")
async def debug_element(data: FrameInput):
    """Debug endpoint to inspect available methods on anaStruct elements"""
    try:
        data = sort_frame_data(data)
        ss = SystemElements()

        # Add one element
        element_id = ss.add_element(location=[data.nodes[0], data.nodes[1]])
        
        # Add a simple support and load for testing
        node_id_map = build_id_maps(ss, data)
        nid_0 = node_id_map[0]
        nid_1 = node_id_map[1]
        
        ss.add_support_hinged(nid_0)
        ss.add_support_roll(nid_1, direction='x')
        ss.point_load(Fy=-10, node_id=nid_0)
        ss.solve()

        element = ss.elements[element_id]
        
        # Get all methods and attributes
        all_attrs = dir(element)
        methods = [attr for attr in all_attrs if callable(getattr(element, attr)) and not attr.startswith('_')]
        
        # Try to get results
        element_results = ss.get_element_results(element_id=0)
        
        return {
            "element_methods": methods,
            "element_results_keys": list(element_results[element_id - 1].keys()) if element_results else [],
            "sample_result": element_results[element_id - 1] if element_results else {},
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}







