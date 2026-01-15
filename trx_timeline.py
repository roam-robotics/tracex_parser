#!/usr/bin/env python3
"""
TraceX Timeline Visualizer

Displays a visual timeline/sequence diagram showing thread execution,
state changes, and events from a TraceX .trx file.
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import numpy as np

from tracex_parser.file_parser import parse_tracex_buffer
from tracex_parser.events import TraceXEvent

# Distinct color palette for threads
THREAD_COLORS = [
    '#e41a1c',  # red
    '#377eb8',  # blue
    '#4daf4a',  # green
    '#984ea3',  # purple
    '#ff7f00',  # orange
    '#ffff33',  # yellow
    '#a65628',  # brown
    '#f781bf',  # pink
    '#999999',  # gray
    '#66c2a5',  # teal
    '#fc8d62',  # salmon
    '#8da0cb',  # periwinkle
    '#e78ac3',  # magenta
    '#a6d854',  # lime
    '#ffd92f',  # gold
]

# Event types that indicate scheduling
RESUME_EVENT_IDS = {1}  # threadResume
SUSPEND_EVENT_IDS = {2}  # threadSuspend
ISR_ENTER_IDS = {3}  # isrEnter
ISR_EXIT_IDS = {4}  # isrExit
RUNNING_IDS = {6}  # running


@dataclass
class ThreadSpan:
    """Represents a period of time when a thread was active/running"""
    thread_name: str
    start_time: int
    end_time: Optional[int] = None
    events: List[TraceXEvent] = field(default_factory=list)


@dataclass
class ThreadInfo:
    """Information about a thread gathered from the trace"""
    name: str
    first_seen: int
    last_seen: int
    total_active_time: int = 0
    spans: List[ThreadSpan] = field(default_factory=list)
    events: List[TraceXEvent] = field(default_factory=list)


def get_thread_name(event: TraceXEvent) -> str:
    """Get a display name for the thread that generated this event"""
    if event.thread_name:
        return event.thread_name
    if event.thread_ptr == 0xFFFFFFFF:
        return "INTERRUPT"
    if event.thread_ptr == 0xF0F0F0F0:
        return "INITIALIZATION"
    return f"Thread_{hex(event.thread_ptr)}"


def analyze_events(events: List[TraceXEvent]) -> Dict[str, ThreadInfo]:
    """Analyze events to extract thread execution information"""
    threads: Dict[str, ThreadInfo] = {}
    active_spans: Dict[str, ThreadSpan] = {}
    
    for event in events:
        thread_name = get_thread_name(event)
        timestamp = event.timestamp
        
        # Initialize thread info if not seen before
        if thread_name not in threads:
            threads[thread_name] = ThreadInfo(
                name=thread_name,
                first_seen=timestamp,
                last_seen=timestamp,
                events=[]
            )
        
        thread_info = threads[thread_name]
        thread_info.last_seen = max(thread_info.last_seen, timestamp)
        thread_info.events.append(event)
        
        # Track active spans based on resume/suspend events
        if event.id in RESUME_EVENT_IDS:
            # Thread is resuming - start a new span
            target_thread = event.mapped_args.get('thread_ptr', thread_name)
            if isinstance(target_thread, str):
                if target_thread not in active_spans:
                    span = ThreadSpan(thread_name=target_thread, start_time=timestamp)
                    active_spans[target_thread] = span
        
        elif event.id in SUSPEND_EVENT_IDS:
            # Thread is suspending - end current span
            target_thread = event.mapped_args.get('thread_ptr', thread_name)
            if isinstance(target_thread, str) and target_thread in active_spans:
                span = active_spans.pop(target_thread)
                span.end_time = timestamp
                if target_thread in threads:
                    threads[target_thread].spans.append(span)
                    threads[target_thread].total_active_time += timestamp - span.start_time
    
    # Close any remaining active spans
    if events:
        final_time = max(e.timestamp for e in events)
        for thread_name, span in active_spans.items():
            span.end_time = final_time
            if thread_name in threads:
                threads[thread_name].spans.append(span)
    
    return threads


def create_timeline_figure(events: List[TraceXEvent], threads: Dict[str, ThreadInfo], 
                          title: str = "TraceX Timeline") -> plt.Figure:
    """Create a matplotlib figure showing the timeline"""
    
    # Sort threads by first seen time
    sorted_threads = sorted(threads.values(), key=lambda t: t.first_seen)
    thread_names = [t.name for t in sorted_threads]
    
    # Assign colors to threads
    thread_colors = {name: THREAD_COLORS[i % len(THREAD_COLORS)] 
                     for i, name in enumerate(thread_names)}
    
    # Calculate time range
    all_times = [e.timestamp for e in events]
    min_time = min(all_times)
    max_time = max(all_times)
    time_range = max_time - min_time if max_time > min_time else 1
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Main timeline subplot
    ax_timeline = fig.add_axes([0.12, 0.35, 0.82, 0.55])
    ax_timeline.set_facecolor('#16213e')
    
    # Event density subplot  
    ax_density = fig.add_axes([0.12, 0.08, 0.82, 0.2])
    ax_density.set_facecolor('#16213e')
    
    # --- Draw Timeline ---
    y_positions = {name: i for i, name in enumerate(thread_names)}
    bar_height = 0.6
    
    # Draw thread execution spans
    for thread_info in sorted_threads:
        y = y_positions[thread_info.name]
        color = thread_colors[thread_info.name]
        
        for span in thread_info.spans:
            if span.end_time is not None:
                width = span.end_time - span.start_time
                rect = Rectangle(
                    (span.start_time, y - bar_height/2),
                    width, bar_height,
                    facecolor=color, edgecolor='white',
                    alpha=0.7, linewidth=0.5
                )
                ax_timeline.add_patch(rect)
    
    # Draw event markers
    event_markers = defaultdict(list)
    for event in events:
        thread_name = get_thread_name(event)
        if thread_name in y_positions:
            event_markers[thread_name].append(event)
    
    for thread_name, thread_events in event_markers.items():
        y = y_positions[thread_name]
        color = thread_colors[thread_name]
        
        for event in thread_events:
            # Different marker styles for different event types
            marker = 'o'
            size = 15
            edge_color = 'white'
            
            if event.id in RESUME_EVENT_IDS:
                marker = '>'
                size = 40
                edge_color = '#00ff00'
            elif event.id in SUSPEND_EVENT_IDS:
                marker = 's'
                size = 40
                edge_color = '#ff6b6b'
            elif event.id in ISR_ENTER_IDS or event.id in ISR_EXIT_IDS:
                marker = '^'
                size = 50
                edge_color = '#ffd93d'
            
            ax_timeline.scatter(
                event.timestamp, y,
                c=[color], s=size, marker=marker,
                edgecolors=edge_color, linewidths=0.5,
                alpha=0.9, zorder=10
            )
    
    # Configure timeline axes
    ax_timeline.set_xlim(min_time - time_range * 0.02, max_time + time_range * 0.02)
    ax_timeline.set_ylim(-0.5, len(thread_names) - 0.5)
    ax_timeline.set_yticks(range(len(thread_names)))
    ax_timeline.set_yticklabels(thread_names, fontsize=10, color='#e8e8e8', 
                                fontfamily='monospace')
    ax_timeline.tick_params(axis='x', colors='#e8e8e8')
    ax_timeline.set_xlabel('Timestamp (ticks)', fontsize=11, color='#e8e8e8')
    ax_timeline.set_title(title, fontsize=14, color='#e94560', fontweight='bold', pad=15)
    
    # Add grid
    ax_timeline.grid(True, axis='x', alpha=0.2, color='#e8e8e8', linestyle='--')
    ax_timeline.axhline(y=-0.5, color='#e8e8e8', alpha=0.3)
    for i in range(len(thread_names)):
        ax_timeline.axhline(y=i + 0.5, color='#e8e8e8', alpha=0.1)
    
    # Style spines
    for spine in ax_timeline.spines.values():
        spine.set_color('#e8e8e8')
        spine.set_alpha(0.3)
    
    # --- Draw Event Density ---
    timestamps = [e.timestamp for e in events]
    bins = min(100, len(set(timestamps)))
    
    ax_density.hist(timestamps, bins=bins, color='#e94560', alpha=0.7, 
                    edgecolor='#e94560', linewidth=0.5)
    ax_density.set_xlim(min_time - time_range * 0.02, max_time + time_range * 0.02)
    ax_density.set_xlabel('Timestamp (ticks)', fontsize=10, color='#e8e8e8')
    ax_density.set_ylabel('Events', fontsize=10, color='#e8e8e8')
    ax_density.tick_params(axis='both', colors='#e8e8e8')
    ax_density.set_title('Event Density', fontsize=11, color='#e8e8e8', pad=8)
    
    for spine in ax_density.spines.values():
        spine.set_color('#e8e8e8')
        spine.set_alpha(0.3)
    
    # --- Add Legend ---
    legend_elements = [
        mpatches.Patch(facecolor=thread_colors[name], edgecolor='white', 
                       alpha=0.7, label=name)
        for name in thread_names[:8]  # Limit legend entries
    ]
    if len(thread_names) > 8:
        legend_elements.append(mpatches.Patch(facecolor='gray', alpha=0.5, 
                                              label=f'... +{len(thread_names)-8} more'))
    
    # Event type legend
    legend_elements.extend([
        plt.Line2D([0], [0], marker='>', color='w', markerfacecolor='gray', 
                   markeredgecolor='#00ff00', markersize=8, label='Resume', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markeredgecolor='#ff6b6b', markersize=8, label='Suspend', linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                   markeredgecolor='#ffd93d', markersize=8, label='ISR', linestyle='None'),
    ])
    
    ax_timeline.legend(handles=legend_elements, loc='upper right', 
                      facecolor='#16213e', edgecolor='#e8e8e8',
                      labelcolor='#e8e8e8', fontsize=8, ncol=2)
    
    return fig


def print_summary(events: List[TraceXEvent], threads: Dict[str, ThreadInfo]):
    """Print a text summary of the trace"""
    print("\n" + "=" * 70)
    print("TRACEX TIMELINE SUMMARY")
    print("=" * 70)
    
    all_times = [e.timestamp for e in events]
    min_time = min(all_times)
    max_time = max(all_times)
    
    print(f"\nTotal Events: {len(events)}")
    print(f"Time Range: {min_time} - {max_time} ({max_time - min_time} ticks)")
    print(f"Unique Threads/Contexts: {len(threads)}")
    
    print("\n" + "-" * 70)
    print("THREAD ACTIVITY")
    print("-" * 70)
    print(f"{'Thread Name':<30} {'Events':>8} {'First Seen':>12} {'Last Seen':>12}")
    print("-" * 70)
    
    for thread in sorted(threads.values(), key=lambda t: t.first_seen):
        print(f"{thread.name:<30} {len(thread.events):>8} {thread.first_seen:>12} {thread.last_seen:>12}")
    
    print("\n" + "-" * 70)
    print("EVENT TYPE BREAKDOWN")
    print("-" * 70)
    
    event_counts = defaultdict(int)
    for event in events:
        event_name = event.fn_name if event.fn_name else f"TX_ID#{event.id}"
        event_counts[event_name] += 1
    
    for event_name, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"  {event_name:<40} {count:>6}")
    
    print("\n" + "-" * 70)
    print("EVENT SEQUENCE (first 30 events)")
    print("-" * 70)
    
    for i, event in enumerate(events[:30]):
        thread_name = get_thread_name(event)
        event_name = event.fn_name if event.fn_name else f"<TX#{event.id}>"
        print(f"{event.timestamp:>10} | {thread_name:<20} | {event_name}")
    
    if len(events) > 30:
        print(f"... and {len(events) - 30} more events")
    
    print("=" * 70 + "\n")


def create_sequence_diagram(events: List[TraceXEvent], threads: Dict[str, ThreadInfo]) -> str:
    """Create a text-based sequence diagram"""
    
    # Get unique thread names in order of first appearance
    thread_order = []
    for event in events:
        name = get_thread_name(event)
        if name not in thread_order:
            thread_order.append(name)
    
    # Create column headers
    col_width = 18
    lines = []
    
    # Header
    header = "    TIME    |"
    for name in thread_order:
        short_name = name[:col_width-2] if len(name) > col_width-2 else name
        header += f" {short_name:^{col_width-1}}|"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Events
    prev_thread = None
    for event in events[:50]:  # Limit to first 50 events
        thread_name = get_thread_name(event)
        event_name = event.fn_name if event.fn_name else f"ID#{event.id}"
        event_name = event_name[:10] if len(event_name) > 10 else event_name
        
        row = f" {event.timestamp:>10} |"
        for name in thread_order:
            if name == thread_name:
                # Show activity in this column
                marker = f"[{event_name}]"
                row += f" {marker:^{col_width-1}}|"
            else:
                # Show connection line or empty
                if prev_thread == name:
                    row += f" {'│':^{col_width-1}}|"
                else:
                    row += f" {'·':^{col_width-1}}|"
        lines.append(row)
        prev_thread = thread_name
    
    if len(events) > 50:
        lines.append(f"... and {len(events) - 50} more events")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TraceX .trx files as a timeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s trace.trx                    # Show timeline and save as PNG
  %(prog)s trace.trx --no-gui           # Text output only
  %(prog)s trace.trx -o my_trace.png    # Save to specific file
  %(prog)s trace.trx --sequence         # Show text sequence diagram
        """
    )
    parser.add_argument('input_trx', help='Path to the input .trx file')
    parser.add_argument('-o', '--output', help='Output image file path (default: <input>_timeline.png)')
    parser.add_argument('--no-gui', action='store_true', help='Disable graphical output (text only)')
    parser.add_argument('--no-save', action='store_true', help='Show plot without saving to file')
    parser.add_argument('--sequence', action='store_true', help='Print text-based sequence diagram')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress text summary output')
    
    args = parser.parse_args()
    
    print(f"Loading TraceX file: {args.input_trx}")
    
    try:
        events, obj_reg_map = parse_tracex_buffer(args.input_trx)
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not events:
        print("No events found in trace file.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(events)} events")
    
    # Analyze events
    threads = analyze_events(events)
    
    # Print summary
    if not args.quiet:
        print_summary(events, threads)
    
    # Print sequence diagram if requested
    if args.sequence:
        print("\nSEQUENCE DIAGRAM")
        print("=" * 70)
        print(create_sequence_diagram(events, threads))
    
    # Create visualization
    if not args.no_gui:
        # Extract filename for title
        title = args.input_trx.split('/')[-1].replace('.trx', '')
        
        fig = create_timeline_figure(events, threads, title=f"TraceX: {title}")
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base_name = args.input_trx.rsplit('.', 1)[0]
            output_path = f"{base_name}_timeline.png"
        
        if not args.no_save:
            fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), 
                       edgecolor='none', bbox_inches='tight')
            print(f"\nTimeline saved to: {output_path}")
        
        plt.show()
    
    print("\nDone!")


if __name__ == '__main__':
    main()

