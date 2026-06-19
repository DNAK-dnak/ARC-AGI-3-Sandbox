# ============================================================
# PATHFINDER Agent — Stages 1-3
#
# Stage 1: Entity Extraction + Persistence Tracking
# Stage 2: Agent Identification
# Stage 3: Action Mapping
#
# Design principles:
#   - Explicit variable names (no single-letter abbreviations)
#   - Every function has a docstring explaining what/why
#   - Test helpers included for each stage
#   - No neural networks — pure observation and logic
# ============================================================

import hashlib
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# STAGE 1: Entity Extraction + Persistence Tracking
# ============================================================

@dataclass
class Entity:
    """A single coherent object in the game frame.
    
    An entity is a connected group of same-colored pixels that
    are not the background color. Think of it as "one object 
    you can see on screen."
    """
    entity_id: int                    # Stable ID that persists across frames
    color: int                        # Color index (0-15)
    pixels: Set[Tuple[int, int]]      # Set of (row, col) coordinates
    pixel_count: int                  # Number of pixels (len(pixels))
    centroid_row: float               # Center of mass, row
    centroid_col: float               # Center of mass, column
    bbox_top: int                     # Bounding box: top row
    bbox_left: int                    # Bounding box: left col
    bbox_bottom: int                  # Bounding box: bottom row
    bbox_right: int                   # Bounding box: right col

    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.centroid_row, self.centroid_col)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.bbox_top, self.bbox_left, self.bbox_bottom, self.bbox_right)

    @property 
    def width(self) -> int:
        return self.bbox_right - self.bbox_left + 1

    @property
    def height(self) -> int:
        return self.bbox_bottom - self.bbox_top + 1

    def shape_signature(self) -> int:
        """Position-invariant hash of the entity's shape.
        
        Two entities with the same shape but different positions 
        will have the same signature. Useful for matching entities
        across frames where they may have moved.
        """
        min_row = self.bbox_top
        min_col = self.bbox_left
        normalized = tuple(sorted(
            (row - min_row, col - min_col) 
            for row, col in self.pixels
        ))
        return hash(normalized)

    def distance_to(self, other: 'Entity') -> float:
        """Manhattan distance between centroids."""
        return (abs(self.centroid_row - other.centroid_row) 
                + abs(self.centroid_col - other.centroid_col))

    def overlaps_with(self, other: 'Entity') -> bool:
        """Check if bounding boxes overlap."""
        return not (
            self.bbox_bottom < other.bbox_top or
            other.bbox_bottom < self.bbox_top or
            self.bbox_right < other.bbox_left or
            other.bbox_right < self.bbox_left
        )

    def is_adjacent_to(self, other: 'Entity', gap: int = 2) -> bool:
        """Check if bounding boxes are within 'gap' pixels."""
        expanded = (
            self.bbox_top - gap, self.bbox_left - gap,
            self.bbox_bottom + gap, self.bbox_right + gap
        )
        return not (
            expanded[2] < other.bbox_top or
            other.bbox_bottom < expanded[0] or
            expanded[3] < other.bbox_left or
            other.bbox_right < expanded[1]
        )

    def __repr__(self):
        return (f"Entity(id={self.entity_id}, color={self.color}, "
                f"pixels={self.pixel_count}, center=({self.centroid_row:.0f},{self.centroid_col:.0f}))")


def detect_background_color(frame: np.ndarray) -> int:
    """Find the background color (most frequent color in the frame).
    
    Args:
        frame: 64x64 array of color indices (0-15)
    
    Returns:
        The color index that appears most frequently.
    """
    color_counts = np.bincount(frame.flatten(), minlength=16)
    return int(color_counts.argmax())


def extract_entities(frame: np.ndarray, background_color: int = -1) -> List[Entity]:
    """Extract all entities from a single frame.
    
    An entity is a connected component of non-background pixels 
    with the same color. Uses 4-connected flood fill (up/down/left/right,
    not diagonal).
    
    Args:
        frame: 64x64 array of color indices (0-15)
        background_color: which color to ignore. If -1, auto-detect.
    
    Returns:
        List of Entity objects, one per connected component.
    """
    height, width = frame.shape
    
    if background_color < 0:
        background_color = detect_background_color(frame)
    
    visited = np.zeros((height, width), dtype=bool)
    entities = []
    next_entity_id = 0
    
    # 4-connected neighbors (up, down, left, right)
    NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for row in range(height):
        for col in range(width):
            # Skip already-visited pixels and background
            if visited[row, col]:
                continue
            if frame[row, col] == background_color:
                visited[row, col] = True
                continue
            
            # Found an unvisited non-background pixel.
            # Flood-fill to find the full connected component.
            color = int(frame[row, col])
            component_pixels = set()
            queue = deque([(row, col)])
            visited[row, col] = True
            
            while queue:
                current_row, current_col = queue.popleft()
                component_pixels.add((current_row, current_col))
                
                for delta_row, delta_col in NEIGHBORS:
                    neighbor_row = current_row + delta_row
                    neighbor_col = current_col + delta_col
                    
                    if (0 <= neighbor_row < height 
                        and 0 <= neighbor_col < width
                        and not visited[neighbor_row, neighbor_col]
                        and frame[neighbor_row, neighbor_col] == color):
                        visited[neighbor_row, neighbor_col] = True
                        queue.append((neighbor_row, neighbor_col))
            
            # Build the Entity from the component
            if len(component_pixels) >= 1:
                rows = [p[0] for p in component_pixels]
                cols = [p[1] for p in component_pixels]
                
                entity = Entity(
                    entity_id=next_entity_id,
                    color=color,
                    pixels=component_pixels,
                    pixel_count=len(component_pixels),
                    centroid_row=sum(rows) / len(rows),
                    centroid_col=sum(cols) / len(cols),
                    bbox_top=min(rows),
                    bbox_left=min(cols),
                    bbox_bottom=max(rows),
                    bbox_right=max(cols),
                )
                entities.append(entity)
                next_entity_id += 1
    
    return entities


# --- Entity Diffs (what changed between two frames) ---

@dataclass
class EntityChange:
    """A single change detected between two frames.
    
    Each change has a type and details about what happened.
    This is the atomic unit of "transformation" — the building
    block for understanding what actions DO.
    """
    change_type: str          # "moved", "appeared", "vanished", 
                              # "recolored", "resized", "reshaped"
    entity_id: int            # Which entity changed
    color: int                # Color of the entity (before change)
    details: Dict             # Type-specific details

    def __repr__(self):
        return f"Change({self.change_type}, id={self.entity_id}, color={self.color}, {self.details})"


@dataclass
class FrameDiff:
    """All changes between two consecutive frames.
    
    This is the complete "what happened" after one action.
    """
    changes: List[EntityChange] = field(default_factory=list)
    frame_changed: bool = False       # Any pixel-level change at all
    entity_config_changed: bool = False  # Entity-level meaningful change
    
    @property
    def has_movement(self) -> bool:
        return any(c.change_type == "moved" for c in self.changes)
    
    @property
    def has_interaction(self) -> bool:
        """Something more interesting than just movement happened."""
        return any(c.change_type in ("appeared", "vanished", "recolored", "resized", "reshaped")
                   for c in self.changes)
    
    @property
    def moved_entity_ids(self) -> List[int]:
        return [c.entity_id for c in self.changes if c.change_type == "moved"]

    def summary(self) -> str:
        if not self.changes:
            return "no change"
        parts = []
        for c in self.changes:
            if c.change_type == "moved":
                parts.append(f"entity {c.entity_id} (color {c.color}) moved ({c.details.get('delta_row',0):+d}, {c.details.get('delta_col',0):+d})")
            elif c.change_type == "appeared":
                parts.append(f"entity {c.entity_id} (color {c.color}) appeared at ({c.details.get('row',0)}, {c.details.get('col',0)})")
            elif c.change_type == "vanished":
                parts.append(f"entity {c.entity_id} (color {c.color}) vanished from ({c.details.get('row',0)}, {c.details.get('col',0)})")
            elif c.change_type == "recolored":
                parts.append(f"entity {c.entity_id} recolored {c.color} → {c.details.get('new_color','?')}")
            elif c.change_type == "resized":
                parts.append(f"entity {c.entity_id} (color {c.color}) resized by {c.details.get('delta_pixels',0):+d} pixels")
            elif c.change_type == "reshaped":
                parts.append(f"entity {c.entity_id} (color {c.color}) reshaped")
        return "; ".join(parts)


def compute_frame_diff(
    previous_entities: List[Entity],
    current_entities: List[Entity],
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
) -> FrameDiff:
    """Compare two frames at the entity level and return all changes.
    
    Matches entities between frames using (color, nearest centroid, similar size).
    Unmatched previous entities → vanished. Unmatched current entities → appeared.
    
    Args:
        previous_entities: entities from the previous frame
        current_entities: entities from the current frame
        previous_frame: raw pixel array of previous frame
        current_frame: raw pixel array of current frame
    
    Returns:
        FrameDiff describing all entity-level changes.
    """
    diff = FrameDiff()
    diff.frame_changed = not np.array_equal(previous_frame, current_frame)
    
    if not diff.frame_changed:
        return diff  # Nothing changed at all
    
    changes = []
    
    # Group entities by color for matching
    prev_by_color: Dict[int, List[Entity]] = {}
    curr_by_color: Dict[int, List[Entity]] = {}
    for entity in previous_entities:
        prev_by_color.setdefault(entity.color, []).append(entity)
    for entity in current_entities:
        curr_by_color.setdefault(entity.color, []).append(entity)
    
    all_colors = set(prev_by_color.keys()) | set(curr_by_color.keys())
    matched_prev_ids = set()
    matched_curr_indices = set()  # indices into curr_by_color[color]
    
    # --- Match entities within the same color group ---
    for color in all_colors:
        prev_list = prev_by_color.get(color, [])
        curr_list = curr_by_color.get(color, [])
        
        # Greedy nearest-centroid matching
        for prev_entity in prev_list:
            best_match_index = None
            best_distance = float('inf')
            
            for idx, curr_entity in enumerate(curr_list):
                # Create a unique key for this color+index combo
                curr_key = (color, idx)
                if curr_key in matched_curr_indices:
                    continue
                
                distance = prev_entity.distance_to(curr_entity)
                # Allow matching if reasonably close (within 32 pixels)
                if distance < best_distance and distance < 32:
                    best_distance = distance
                    best_match_index = idx
            
            if best_match_index is not None:
                curr_entity = curr_list[best_match_index]
                matched_prev_ids.add(prev_entity.entity_id)
                matched_curr_indices.add((color, best_match_index))
                
                # Detect what changed about this entity
                delta_row = curr_entity.centroid_row - prev_entity.centroid_row
                delta_col = curr_entity.centroid_col - prev_entity.centroid_col
                
                # Movement detection (threshold: 0.5 pixel)
                if abs(delta_row) > 0.5 or abs(delta_col) > 0.5:
                    changes.append(EntityChange(
                        change_type="moved",
                        entity_id=prev_entity.entity_id,
                        color=color,
                        details={
                            "delta_row": round(delta_row),
                            "delta_col": round(delta_col),
                            "new_row": round(curr_entity.centroid_row),
                            "new_col": round(curr_entity.centroid_col),
                        }
                    ))
                
                # Size change detection
                size_delta = curr_entity.pixel_count - prev_entity.pixel_count
                if size_delta != 0:
                    changes.append(EntityChange(
                        change_type="resized",
                        entity_id=prev_entity.entity_id,
                        color=color,
                        details={"delta_pixels": size_delta}
                    ))
                
                # Shape change detection (same size, different shape)
                if (size_delta == 0 
                    and curr_entity.shape_signature() != prev_entity.shape_signature()):
                    changes.append(EntityChange(
                        change_type="reshaped",
                        entity_id=prev_entity.entity_id,
                        color=color,
                        details={}
                    ))
    
    # --- Detect vanished entities (unmatched previous) ---
    for color, prev_list in prev_by_color.items():
        for prev_entity in prev_list:
            if prev_entity.entity_id not in matched_prev_ids:
                changes.append(EntityChange(
                    change_type="vanished",
                    entity_id=prev_entity.entity_id,
                    color=color,
                    details={
                        "row": round(prev_entity.centroid_row),
                        "col": round(prev_entity.centroid_col),
                        "pixel_count": prev_entity.pixel_count,
                    }
                ))
    
    # --- Detect appeared entities (unmatched current) ---
    for color, curr_list in curr_by_color.items():
        for idx, curr_entity in enumerate(curr_list):
            if (color, idx) not in matched_curr_indices:
                changes.append(EntityChange(
                    change_type="appeared",
                    entity_id=curr_entity.entity_id,
                    color=color,
                    details={
                        "row": round(curr_entity.centroid_row),
                        "col": round(curr_entity.centroid_col),
                        "pixel_count": curr_entity.pixel_count,
                    }
                ))
    
    # --- Detect recoloring (entity at same position changed color) ---
    for prev_entity in previous_entities:
        if prev_entity.entity_id in matched_prev_ids:
            continue  # Already matched within same color
        for curr_entity in current_entities:
            if curr_entity.color == prev_entity.color:
                continue  # Same color, not a recolor
            distance = prev_entity.distance_to(curr_entity)
            size_ratio = min(prev_entity.pixel_count, curr_entity.pixel_count) / max(prev_entity.pixel_count, curr_entity.pixel_count, 1)
            # Close position + similar size = likely recolored
            if distance < 3 and size_ratio > 0.7:
                changes.append(EntityChange(
                    change_type="recolored",
                    entity_id=prev_entity.entity_id,
                    color=prev_entity.color,
                    details={
                        "new_color": curr_entity.color,
                        "position": (round(prev_entity.centroid_row), round(prev_entity.centroid_col)),
                    }
                ))
    
    diff.changes = changes
    diff.entity_config_changed = len(changes) > 0
    return diff


# ============================================================
# STAGE 2: Agent Identification
# ============================================================

@dataclass
class AgentIdentity:
    """The result of identifying which entity the player controls.
    
    If agent_entity_id is None, this is a click-only game 
    (no controllable avatar).
    """
    agent_entity_id: Optional[int] = None
    agent_color: Optional[int] = None
    game_type: str = "unknown"  # "movement", "click", "hybrid", "unknown"
    confidence: float = 0.0     # 0.0 to 1.0
    movement_evidence: Dict[int, List[int]] = field(default_factory=dict)
    # movement_evidence[action_id] = [entity_ids that moved]

    def __repr__(self):
        if self.agent_entity_id is not None:
            return (f"AgentIdentity(entity={self.agent_entity_id}, "
                    f"color={self.agent_color}, type={self.game_type}, "
                    f"confidence={self.confidence:.2f})")
        return f"AgentIdentity(no agent, type={self.game_type})"


def identify_agent(action_diffs: Dict[int, FrameDiff]) -> AgentIdentity:
    """Determine which entity (if any) the player controls.
    
    Logic: 
    - Take the frame diffs from trying each action (ACTION1-5).
    - For each action, note which entities moved.
    - The entity that moved for the MOST DIFFERENT actions is the agent.
    - If no entity ever moved → click-only game.
    
    Args:
        action_diffs: dict mapping action_id (1-5) to the FrameDiff 
                      that resulted from taking that action.
    
    Returns:
        AgentIdentity with the identified agent (or None for click games).
    """
    result = AgentIdentity()
    
    # Track which entities moved for each action
    entity_move_count: Dict[int, int] = {}  # entity_id → number of actions that moved it
    entity_colors: Dict[int, int] = {}       # entity_id → color
    
    for action_id, diff in action_diffs.items():
        moved_ids = []
        for change in diff.changes:
            if change.change_type == "moved":
                moved_ids.append(change.entity_id)
                entity_move_count[change.entity_id] = entity_move_count.get(change.entity_id, 0) + 1
                entity_colors[change.entity_id] = change.color
        result.movement_evidence[action_id] = moved_ids
    
    if not entity_move_count:
        # No entity ever moved for any key action → click-only game
        result.game_type = "click"
        result.confidence = 0.8
        return result
    
    # Find the entity that moved for the most different actions
    best_entity_id = max(entity_move_count, key=entity_move_count.get)
    actions_that_moved_it = entity_move_count[best_entity_id]
    total_actions_tried = len(action_diffs)
    
    result.agent_entity_id = best_entity_id
    result.agent_color = entity_colors.get(best_entity_id)
    
    # Confidence: how many actions moved this entity vs how many we tried
    result.confidence = actions_that_moved_it / max(total_actions_tried, 1)
    
    if result.confidence >= 0.5:
        result.game_type = "movement"
    elif result.confidence > 0:
        result.game_type = "hybrid"  # Some actions move, some don't
    
    return result


# ============================================================
# STAGE 3: Action Mapping
# ============================================================

@dataclass
class ActionEffect:
    """What a single action does, learned from observation.
    
    This represents our understanding of one button press.
    """
    action_id: int
    effect_type: str       # "move", "interact", "noop", "context_dependent"
    direction: Optional[Tuple[int, int]] = None  # (delta_row, delta_col) for movement
    moves_agent: bool = False
    changes_other_entities: bool = False
    observation_count: int = 1  # How many times we've observed this
    
    def __repr__(self):
        if self.effect_type == "move" and self.direction:
            direction_name = {
                (-1, 0): "up", (1, 0): "down",
                (0, -1): "left", (0, 1): "right",
                (-1, -1): "up-left", (-1, 1): "up-right",
                (1, -1): "down-left", (1, 1): "down-right",
            }.get(self.direction, f"({self.direction[0]:+d},{self.direction[1]:+d})")
            return f"Action {self.action_id}: move {direction_name}"
        return f"Action {self.action_id}: {self.effect_type}"


@dataclass
class ActionMap:
    """Complete map of what each action does.
    
    Built from the same observations used for agent identification.
    """
    actions: Dict[int, ActionEffect] = field(default_factory=dict)
    available_action_ids: List[int] = field(default_factory=list)
    has_click_action: bool = False
    
    def movement_actions(self) -> List[ActionEffect]:
        """Actions that move the agent."""
        return [a for a in self.actions.values() if a.effect_type == "move"]
    
    def interaction_actions(self) -> List[ActionEffect]:
        """Actions that do something other than movement."""
        return [a for a in self.actions.values() if a.effect_type == "interact"]
    
    def noop_actions(self) -> List[ActionEffect]:
        """Actions that had no visible effect."""
        return [a for a in self.actions.values() if a.effect_type == "noop"]
    
    def get_direction_action(self, delta_row: int, delta_col: int) -> Optional[int]:
        """Find which action moves the agent in a given direction."""
        target = (delta_row, delta_col)
        for action in self.actions.values():
            if action.effect_type == "move" and action.direction == target:
                return action.action_id
        return None

    def summary(self) -> str:
        lines = [f"ActionMap ({len(self.actions)} actions mapped, click={self.has_click_action}):"]
        for action_id in sorted(self.actions.keys()):
            lines.append(f"  {self.actions[action_id]}")
        return "\n".join(lines)


def build_action_map(
    action_diffs: Dict[int, FrameDiff],
    agent_identity: AgentIdentity,
    available_action_ids: List[int],
) -> ActionMap:
    """Build a map of what each action does.
    
    Uses the frame diffs from probing (same data as agent identification)
    plus the agent identity to classify each action.
    
    Args:
        action_diffs: dict mapping action_id (1-5) to FrameDiff
        agent_identity: result from identify_agent()
        available_action_ids: all action IDs the game supports (1-7)
    
    Returns:
        ActionMap describing each action's effect.
    """
    action_map = ActionMap()
    action_map.available_action_ids = available_action_ids
    action_map.has_click_action = 6 in available_action_ids
    
    agent_id = agent_identity.agent_entity_id
    
    for action_id, diff in action_diffs.items():
        if not diff.frame_changed:
            # Nothing happened at all
            action_map.actions[action_id] = ActionEffect(
                action_id=action_id,
                effect_type="noop",
            )
            continue
        
        # Check if the agent moved
        agent_moved = False
        agent_direction = None
        other_entities_changed = False
        
        for change in diff.changes:
            if change.entity_id == agent_id and change.change_type == "moved":
                agent_moved = True
                agent_direction = (
                    change.details.get("delta_row", 0),
                    change.details.get("delta_col", 0),
                )
            elif change.entity_id != agent_id:
                other_entities_changed = True
        
        if agent_moved and agent_direction:
            # Normalize direction to unit vector
            dr, dc = agent_direction
            norm_dr = 0 if dr == 0 else (1 if dr > 0 else -1)
            norm_dc = 0 if dc == 0 else (1 if dc > 0 else -1)
            
            action_map.actions[action_id] = ActionEffect(
                action_id=action_id,
                effect_type="move",
                direction=(norm_dr, norm_dc),
                moves_agent=True,
                changes_other_entities=other_entities_changed,
            )
        elif other_entities_changed or diff.has_interaction:
            action_map.actions[action_id] = ActionEffect(
                action_id=action_id,
                effect_type="interact",
                moves_agent=False,
                changes_other_entities=True,
            )
        elif diff.frame_changed and not diff.entity_config_changed:
            # Pixels changed but no entity-level change detected.
            # Could be animation, HUD update, etc.
            action_map.actions[action_id] = ActionEffect(
                action_id=action_id,
                effect_type="noop",  # Visually changed but not meaningfully
            )
        else:
            action_map.actions[action_id] = ActionEffect(
                action_id=action_id,
                effect_type="context_dependent",
                moves_agent=agent_moved,
                changes_other_entities=other_entities_changed,
            )
    
    return action_map


# ============================================================
# STAGE 1-3 COMBINED: The Probing Protocol
# ============================================================

@dataclass
class ProbeResult:
    """Complete result of the initial probing phase (Stages 1-3).
    
    Contains everything the agent has learned from ~5 purposeful actions.
    """
    background_color: int
    initial_entities: List[Entity]
    agent_identity: AgentIdentity
    action_map: ActionMap
    action_diffs: Dict[int, FrameDiff]  # Raw diffs for further analysis
    actions_spent: int                   # How many real actions were used


# ============================================================
# TEST HELPERS
# ============================================================

def make_test_frame(entities_spec: List[dict], bg_color: int = 0, size: int = 64) -> np.ndarray:
    """Create a test frame from a simple entity specification.
    
    Args:
        entities_spec: list of dicts with keys:
            - color: int (1-15)
            - row, col: int (top-left corner)
            - width, height: int (size of rectangle)
        bg_color: background color (default 0)
        size: frame size (default 64)
    
    Returns:
        np.ndarray of shape (size, size) with the specified entities.
    
    Example:
        frame = make_test_frame([
            {"color": 3, "row": 10, "col": 20, "width": 4, "height": 4},
            {"color": 7, "row": 30, "col": 40, "width": 6, "height": 2},
        ])
    """
    frame = np.full((size, size), bg_color, dtype=np.int64)
    for spec in entities_spec:
        r, c = spec["row"], spec["col"]
        h, w = spec["height"], spec["width"]
        frame[r:r+h, c:c+w] = spec["color"]
    return frame


def test_stage1():
    """Test entity extraction and diff computation."""
    print("=" * 60)
    print("STAGE 1 TESTS: Entity Extraction + Persistence")
    print("=" * 60)
    
    # Test 1: Basic extraction
    frame = make_test_frame([
        {"color": 3, "row": 10, "col": 20, "width": 4, "height": 4},  # blue square
        {"color": 7, "row": 30, "col": 40, "width": 6, "height": 2},  # pink rectangle
        {"color": 3, "row": 50, "col": 10, "width": 3, "height": 3},  # another blue square
    ])
    
    bg = detect_background_color(frame)
    entities = extract_entities(frame, bg)
    
    print(f"\nTest 1: Basic extraction")
    print(f"  Background color: {bg}")
    print(f"  Entities found: {len(entities)}")
    for e in entities:
        print(f"    {e}")
    assert len(entities) == 3, f"Expected 3 entities, got {len(entities)}"
    assert bg == 0, f"Expected bg=0, got {bg}"
    print("  PASSED")
    
    # Test 2: Stability (same frame → same entities)
    entities_again = extract_entities(frame, bg)
    assert len(entities) == len(entities_again), "Unstable extraction count"
    for e1, e2 in zip(entities, entities_again):
        assert e1.pixel_count == e2.pixel_count, "Unstable pixel count"
        assert e1.color == e2.color, "Unstable color"
    print(f"\nTest 2: Stability — PASSED")
    
    # Test 3: Movement diff
    frame_moved = make_test_frame([
        {"color": 3, "row": 12, "col": 20, "width": 4, "height": 4},  # moved down 2
        {"color": 7, "row": 30, "col": 40, "width": 6, "height": 2},  # unchanged
        {"color": 3, "row": 50, "col": 10, "width": 3, "height": 3},  # unchanged
    ])
    entities_moved = extract_entities(frame_moved, bg)
    diff = compute_frame_diff(entities, entities_moved, frame, frame_moved)
    
    print(f"\nTest 3: Movement detection")
    print(f"  Frame changed: {diff.frame_changed}")
    print(f"  Changes: {diff.summary()}")
    assert diff.frame_changed, "Should detect frame change"
    assert diff.has_movement, "Should detect movement"
    print("  PASSED")
    
    # Test 4: Vanish + appear
    frame_vanished = make_test_frame([
        {"color": 7, "row": 30, "col": 40, "width": 6, "height": 2},  # only pink remains
    ])
    entities_vanished = extract_entities(frame_vanished, bg)
    diff2 = compute_frame_diff(entities, entities_vanished, frame, frame_vanished)
    
    print(f"\nTest 4: Vanish detection")
    print(f"  Changes: {diff2.summary()}")
    vanish_count = sum(1 for c in diff2.changes if c.change_type == "vanished")
    assert vanish_count == 2, f"Expected 2 vanished, got {vanish_count}"
    print("  PASSED")
    
    # Test 5: Speed
    import time
    large_frame = make_test_frame([
        {"color": c, "row": r*8, "col": c*8, "width": 5, "height": 5}
        for r in range(7) for c in range(1, 8)
    ])
    start = time.time()
    for _ in range(100):
        extract_entities(large_frame)
    elapsed = (time.time() - start) / 100
    print(f"\nTest 5: Speed — {elapsed*1000:.2f}ms per extraction")
    assert elapsed < 0.05, f"Too slow: {elapsed*1000:.1f}ms (target <50ms)"
    print("  PASSED")
    
    print("\nAll Stage 1 tests PASSED")


def test_stage2():
    """Test agent identification with simulated action diffs."""
    print("\n" + "=" * 60)
    print("STAGE 2 TESTS: Agent Identification")
    print("=" * 60)
    
    # Simulate a movement game: entity 0 (blue) moves for actions 1-4
    diffs = {}
    for action_id in range(1, 6):
        diff = FrameDiff(frame_changed=True)
        if action_id <= 4:  # Actions 1-4 move the agent
            diff.changes = [EntityChange(
                change_type="moved", entity_id=0, color=3,
                details={"delta_row": [-1, 1, 0, 0][action_id-1],
                         "delta_col": [0, 0, -1, 1][action_id-1]}
            )]
            diff.entity_config_changed = True
        # Action 5 does nothing
        diffs[action_id] = diff
    
    identity = identify_agent(diffs)
    print(f"\nTest 1: Movement game")
    print(f"  {identity}")
    assert identity.agent_entity_id == 0, f"Expected agent=0, got {identity.agent_entity_id}"
    assert identity.game_type == "movement", f"Expected movement, got {identity.game_type}"
    assert identity.confidence >= 0.5
    print("  PASSED")
    
    # Simulate a click-only game: no entity ever moves for key actions
    diffs_click = {i: FrameDiff(frame_changed=False) for i in range(1, 6)}
    identity_click = identify_agent(diffs_click)
    print(f"\nTest 2: Click-only game")
    print(f"  {identity_click}")
    assert identity_click.agent_entity_id is None
    assert identity_click.game_type == "click"
    print("  PASSED")
    
    print("\nAll Stage 2 tests PASSED")


def test_stage3():
    """Test action mapping with simulated data."""
    print("\n" + "=" * 60)
    print("STAGE 3 TESTS: Action Mapping")
    print("=" * 60)
    
    # Build diffs for a typical movement game
    directions = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    diffs = {}
    for action_id in range(1, 6):
        diff = FrameDiff(frame_changed=action_id <= 4)
        if action_id in directions:
            dr, dc = directions[action_id]
            diff.changes = [EntityChange(
                change_type="moved", entity_id=0, color=3,
                details={"delta_row": dr, "delta_col": dc}
            )]
            diff.entity_config_changed = True
        diffs[action_id] = diff
    
    identity = AgentIdentity(
        agent_entity_id=0, agent_color=3,
        game_type="movement", confidence=0.8
    )
    
    action_map = build_action_map(diffs, identity, [1, 2, 3, 4, 5, 6])
    
    print(f"\nTest 1: Movement game action map")
    print(f"  {action_map.summary()}")
    
    assert len(action_map.movement_actions()) == 4, "Expected 4 movement actions"
    assert len(action_map.noop_actions()) == 1, "Expected 1 noop action"
    assert action_map.has_click_action
    
    # Test direction lookup
    up_action = action_map.get_direction_action(-1, 0)
    assert up_action == 1, f"Expected ACTION1=up, got {up_action}"
    right_action = action_map.get_direction_action(0, 1)
    assert right_action == 4, f"Expected ACTION4=right, got {right_action}"
    
    print("  PASSED")
    
    print("\nAll Stage 3 tests PASSED")


def run_all_tests():
    """Run all stage tests."""
    test_stage1()
    test_stage2()
    test_stage3()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


# Run tests if executed directly
if __name__ == "__main__":
    run_all_tests()