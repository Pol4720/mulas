"""
Exporters Module
================

Export utilities for solutions and reports.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from io import StringIO

import sys
sys.path.insert(0, str(__file__).rsplit('utils', 1)[0])

from core.problem import Problem, Solution, Item, Bin


class SolutionExporter:
    """
    Exports solutions to various formats.
    
    Supported formats:
    - JSON
    - CSV
    - Plain text
    - LaTeX
    - HTML
    """
    
    @staticmethod
    def to_json(solution: Solution, problem: Optional[Problem] = None,
                include_metadata: bool = True) -> str:
        """
        Export solution to JSON format.
        
        Parameters
        ----------
        solution : Solution
            Solution to export
        problem : Problem, optional
            Original problem for metadata
        include_metadata : bool
            Include additional metadata
            
        Returns
        -------
        str
            JSON string
        """
        data = {
            'bins': [
                {
                    'id': f"bin_{i+1}",
                    'items': [
                        {
                            'id': item.id,
                            'weight': item.weight,
                            'value': item.value
                        }
                        for item in bin_obj.items
                    ],
                    'total_weight': sum(item.weight for item in bin_obj.items),
                    'total_value': sum(item.value for item in bin_obj.items)
                }
                for i, bin_obj in enumerate(solution.bins)
            ]
        }
        
        if include_metadata:
            data['metadata'] = {
                'num_bins': len(solution.bins),
                'total_items': sum(len(b.items) for b in solution.bins),
                'objective': SolutionExporter._calculate_objective(solution),
                'exported_at': datetime.now().isoformat()
            }
            
            if problem:
                data['metadata']['problem_name'] = problem.name
                data['metadata']['bin_capacity'] = problem.bin_capacity
        
        return json.dumps(data, indent=2)
    
    @staticmethod
    def to_csv(solution: Solution) -> str:
        """
        Export solution to CSV format.
        
        Parameters
        ----------
        solution : Solution
            Solution to export
            
        Returns
        -------
        str
            CSV string
        """
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['bin_id', 'item_id', 'weight', 'value'])
        
        # Data rows
        for i, bin_obj in enumerate(solution.bins):
            bin_id = f"bin_{i+1}"
            for item in bin_obj.items:
                writer.writerow([bin_id, item.id, item.weight, item.value])
        
        return output.getvalue()
    
    @staticmethod
    def to_text(solution: Solution, problem: Optional[Problem] = None) -> str:
        """
        Export solution to plain text format.
        
        Parameters
        ----------
        solution : Solution
            Solution to export
        problem : Problem, optional
            Original problem for context
            
        Returns
        -------
        str
            Formatted text string
        """
        lines = [
            "=" * 50,
            "SOLUTION REPORT",
            "=" * 50,
            ""
        ]
        
        if problem:
            lines.extend([
                f"Problem: {problem.name}",
                f"Items: {len(problem.items)}",
                f"Bins: {problem.num_bins}",
                f"Capacity: {problem.bin_capacity}",
                ""
            ])
        
        lines.append("-" * 50)
        
        total_weight = 0
        total_value = 0
        bin_values = []
        
        for i, bin_obj in enumerate(solution.bins):
            bin_weight = sum(item.weight for item in bin_obj.items)
            bin_value = sum(item.value for item in bin_obj.items)
            total_weight += bin_weight
            total_value += bin_value
            bin_values.append(bin_value)
            
            capacity = problem.bin_capacity if problem else 100
            utilization = bin_weight / capacity * 100
            
            lines.extend([
                f"Bin {i+1}:",
                f"  Items: {len(bin_obj.items)}",
                f"  Weight: {bin_weight:.2f} / {capacity} ({utilization:.1f}%)",
                f"  Value: {bin_value:.2f}",
                f"  Contents:"
            ])
            
            for item in bin_obj.items:
                lines.append(f"    - {item.id}: w={item.weight:.2f}, v={item.value:.2f}")
            
            lines.append("")
        
        lines.append("-" * 50)
        
        # Summary
        objective = max(bin_values) - min(bin_values) if bin_values else 0
        
        lines.extend([
            "SUMMARY",
            f"  Total Weight: {total_weight:.2f}",
            f"  Total Value: {total_value:.2f}",
            f"  Objective (value range): {objective:.4f}",
            f"  Balance Score: {1 - (max(bin_values) - min(bin_values)) / (sum(bin_values) / len(bin_values)) if bin_values else 0:.4f}",
            "=" * 50
        ])
        
        return '\n'.join(lines)
    
    @staticmethod
    def to_latex(solution: Solution, problem: Optional[Problem] = None) -> str:
        """
        Export solution to LaTeX format.
        
        Parameters
        ----------
        solution : Solution
            Solution to export
        problem : Problem, optional
            Original problem for context
            
        Returns
        -------
        str
            LaTeX string
        """
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Solution Assignment}"
        ]
        
        if problem:
            lines.append(f"\\label{{tab:solution_{problem.name}}}")
        
        lines.extend([
            "\\begin{tabular}{|c|c|c|c|}",
            "\\hline",
            "\\textbf{Bin} & \\textbf{Item} & \\textbf{Weight} & \\textbf{Value} \\\\",
            "\\hline"
        ])
        
        for i, bin_obj in enumerate(solution.bins):
            bin_id = f"Bin {i+1}"
            first_item = True
            
            for item in bin_obj.items:
                if first_item:
                    lines.append(
                        f"{bin_id} & {item.id} & {item.weight:.2f} & {item.value:.2f} \\\\"
                    )
                    first_item = False
                else:
                    lines.append(
                        f" & {item.id} & {item.weight:.2f} & {item.value:.2f} \\\\"
                    )
            
            lines.append("\\hline")
        
        lines.extend([
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        # Add summary table
        bin_values = [sum(item.value for item in b.items) for b in solution.bins]
        bin_weights = [sum(item.weight for item in b.items) for b in solution.bins]
        
        lines.extend([
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Bin Summary}",
            "\\begin{tabular}{|c|c|c|c|}",
            "\\hline",
            "\\textbf{Bin} & \\textbf{Items} & \\textbf{Total Weight} & \\textbf{Total Value} \\\\",
            "\\hline"
        ])
        
        for i, bin_obj in enumerate(solution.bins):
            lines.append(
                f"Bin {i+1} & {len(bin_obj.items)} & {bin_weights[i]:.2f} & {bin_values[i]:.2f} \\\\"
            )
        
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return '\n'.join(lines)
    
    @staticmethod
    def to_html(solution: Solution, problem: Optional[Problem] = None) -> str:
        """
        Export solution to HTML format.
        
        Parameters
        ----------
        solution : Solution
            Solution to export
        problem : Problem, optional
            Original problem for context
            
        Returns
        -------
        str
            HTML string
        """
        bin_values = [sum(item.value for item in b.items) for b in solution.bins]
        objective = max(bin_values) - min(bin_values) if bin_values else 0
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Solution Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                  background: #f5f5f5; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1>Solution Report</h1>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{len(solution.bins)}</div>
            <div class="metric-label">Bins Used</div>
        </div>
        <div class="metric">
            <div class="metric-value">{sum(len(b.items) for b in solution.bins)}</div>
            <div class="metric-label">Items Assigned</div>
        </div>
        <div class="metric">
            <div class="metric-value">{objective:.4f}</div>
            <div class="metric-label">Objective Value</div>
        </div>
    </div>
    
    <h2>Bin Contents</h2>
    <table>
        <tr>
            <th>Bin</th>
            <th>Item</th>
            <th>Weight</th>
            <th>Value</th>
        </tr>
"""
        
        for i, bin_obj in enumerate(solution.bins):
            for j, item in enumerate(bin_obj.items):
                bin_label = f"Bin {i+1}" if j == 0 else ""
                html += f"""        <tr>
            <td>{bin_label}</td>
            <td>{item.id}</td>
            <td>{item.weight:.2f}</td>
            <td>{item.value:.2f}</td>
        </tr>
"""
        
        html += """    </table>
</body>
</html>
"""
        return html
    
    @staticmethod
    def save(solution: Solution, filepath: str, format: str = 'json',
             problem: Optional[Problem] = None):
        """
        Save solution to file.
        
        Parameters
        ----------
        solution : Solution
            Solution to save
        filepath : str
            Output file path
        format : str
            Output format (json, csv, txt, tex, html)
        problem : Problem, optional
            Original problem
        """
        exporters = {
            'json': SolutionExporter.to_json,
            'csv': SolutionExporter.to_csv,
            'txt': SolutionExporter.to_text,
            'tex': SolutionExporter.to_latex,
            'html': SolutionExporter.to_html
        }
        
        if format not in exporters:
            raise ValueError(f"Unknown format: {format}. Supported: {list(exporters.keys())}")
        
        if format in ['json', 'txt', 'tex', 'html']:
            content = exporters[format](solution, problem)
        else:
            content = exporters[format](solution)
        
        with open(filepath, 'w') as f:
            f.write(content)
    
    @staticmethod
    def _calculate_objective(solution: Solution) -> float:
        """Calculate objective value."""
        bin_values = [sum(item.value for item in b.items) for b in solution.bins]
        return max(bin_values) - min(bin_values) if bin_values else 0


class ReportGenerator:
    """
    Generates comprehensive reports for benchmark results.
    """
    
    @staticmethod
    def generate_latex_report(results: Dict[str, Any], 
                             output_path: str,
                             title: str = "Benchmark Results"):
        """
        Generate a LaTeX report from benchmark results.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Benchmark results
        output_path : str
            Output file path
        title : str
            Report title
        """
        content = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\usepackage{{hyperref}}

\\title{{{title}}}
\\author{{Benchmark System}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Introduction}}

This report presents the benchmark results for the Balanced Multi-Bin Packing problem.

\\section{{Methodology}}

The benchmark was conducted with the following configuration:
\\begin{{itemize}}
    \\item Number of runs per instance: {results.get('config', {}).get('num_runs', 'N/A')}
    \\item Time limit per run: {results.get('config', {}).get('time_limit', 'N/A')} seconds
    \\item Random seed: {results.get('config', {}).get('seed', 'N/A')}
\\end{{itemize}}

\\section{{Results}}

\\subsection{{Algorithm Comparison}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Algorithm Performance Summary}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Algorithm & Mean Obj & Std & Best & Feasible\\% & Avg Time \\\\
\\midrule
"""
        
        summary = results.get('summary', {})
        for algo, stats in sorted(summary.items(), key=lambda x: x[1].get('avg_objective', float('inf'))):
            content += f"{algo} & {stats.get('avg_objective', 0):.4f} & "
            content += f"{stats.get('std_objective', 0):.4f} & "
            content += f"{stats.get('best_objective', 0):.4f} & "
            content += f"{stats.get('feasible_rate', 0)*100:.1f}\\% & "
            content += f"{stats.get('avg_time', 0):.3f}s \\\\\n"
        
        content += """\\bottomrule
\\end{tabular}
\\end{table}

\\section{Conclusion}

Based on the benchmark results, the algorithms can be ranked by their average objective value.
Further analysis may be needed to determine statistical significance of the differences.

\\end{document}
"""
        
        with open(output_path, 'w') as f:
            f.write(content)
    
    @staticmethod
    def generate_markdown_report(results: Dict[str, Any]) -> str:
        """
        Generate a Markdown report from benchmark results.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Benchmark results
            
        Returns
        -------
        str
            Markdown content
        """
        lines = [
            "# Benchmark Results Report",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Configuration",
            "",
            f"- **Runs per instance:** {results.get('config', {}).get('num_runs', 'N/A')}",
            f"- **Time limit:** {results.get('config', {}).get('time_limit', 'N/A')}s",
            f"- **Random seed:** {results.get('config', {}).get('seed', 'N/A')}",
            "",
            "## Algorithm Comparison",
            "",
            "| Algorithm | Mean Obj | Std | Best | Feasible% | Avg Time |",
            "|-----------|----------|-----|------|-----------|----------|"
        ]
        
        summary = results.get('summary', {})
        for algo, stats in sorted(summary.items(), key=lambda x: x[1].get('avg_objective', float('inf'))):
            lines.append(
                f"| {algo} | {stats.get('avg_objective', 0):.4f} | "
                f"{stats.get('std_objective', 0):.4f} | "
                f"{stats.get('best_objective', 0):.4f} | "
                f"{stats.get('feasible_rate', 0)*100:.1f}% | "
                f"{stats.get('avg_time', 0):.3f}s |"
            )
        
        lines.extend([
            "",
            "## Conclusion",
            "",
            "The table above shows the performance metrics for each algorithm.",
            "Lower objective values indicate better balance between bins."
        ])
        
        return '\n'.join(lines)
