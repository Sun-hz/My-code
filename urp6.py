import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd


class MatrixAnalyzer:
    def __init__(self, data):
        self.data = data
        self.num_subjects = data.shape[0]
        self.subject_ids = list(range(self.num_subjects))

    def get_subject_matrix(self, subject_id):
        if subject_id not in self.subject_ids:
            raise ValueError("Subject ID does not exist")
        return self.data[subject_id]

    def calculate_features(self, matrix):
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            trace = np.trace(matrix)
            determinant = np.linalg.det(matrix)

            G = nx.from_numpy_array(matrix)
            average_clustering = nx.average_clustering(G)
            degree_centrality = nx.degree_centrality(G)
            average_degree_centrality = np.mean(list(degree_centrality.values()))

            diameter = nx.diameter(G) if nx.is_connected(G) else np.nan
            average_shortest_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan

            return eigenvalues, singular_values, trace, determinant, average_clustering, average_degree_centrality, diameter, average_shortest_path_length
        except Exception as e:
            messagebox.showerror("Error", f"Feature calculation failed: {str(e)}")
            return None, None, None, None, None, None, None, None


class BrainNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Network Analysis Software")
        self.root.geometry("1200x1000")  # 固定初始宽度，高度自适应
        self.analyzer = None
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # 创建垂直滚动条和主Canvas
        self.canvas = tk.Canvas(self.root, bg='#f0f0f0', highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.vsb = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.main_frame = tk.Frame(self.canvas, bg='#f0f0f0')
        self.canvas.create_window((0, 0), window=self.main_frame, anchor=tk.NW)

        self.main_frame.bind("<Configure>", self.on_frame_configure)

        # 界面组件添加到main_frame
        self.add_buttons_and_text(self.main_frame)
        self.add_heatmap_area(self.main_frame)

    def add_buttons_and_text(self, parent):
        self.btn_import_data = tk.Button(
            parent,
            text="Import Data (npy/csv file)",
            command=self.import_data,
            font=('Arial', 14),
            bg='#77c9f0',
            padx=20,
            pady=10
        )
        self.btn_import_data.pack(pady=15, fill=tk.X, padx=30)

        self.btn_simulate = tk.Button(
            parent,
            text="Generate Simulated Data",
            command=self.open_dimension_window,
            font=('Arial', 14),
            bg='#ffbb33',
            padx=20,
            pady=10
        )
        self.btn_simulate.pack(pady=15, fill=tk.X, padx=30)

        self.btn_show_ids = tk.Button(
            parent,
            text="Show All Subject IDs",
            command=self.show_all_subject_ids,
            font=('Arial', 14),
            bg='#e0e0e0',
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.btn_show_ids.pack(pady=10, fill=tk.X, padx=30)

        self.id_frame = tk.Frame(parent, bg='#f0f0f0')
        self.id_label = tk.Label(
            self.id_frame,
            text="Enter Subject ID:",
            font=('Arial', 14),
            bg='#f0f0f0'
        )
        self.id_label.pack(side=tk.LEFT, padx=10)
        self.id_entry = tk.Entry(
            self.id_frame,
            width=20,
            font=('Arial', 14)
        )
        self.id_entry.pack(side=tk.LEFT, padx=10)
        self.id_frame.pack(pady=15)

        self.btn_analyze = tk.Button(
            parent,
            text="Analyze",
            command=self.analyze_subject,
            font=('Arial', 16, 'bold'),
            bg='#4CAF50',
            padx=30,
            pady=15,
            state=tk.DISABLED
        )
        self.btn_analyze.pack(pady=10)

        self.btn_clear = tk.Button(
            parent,
            text="Clear Data",
            command=self.clear_data,
            font=('Arial', 14),
            bg='#ff4444',
            padx=20,
            pady=10
        )
        self.btn_clear.pack(pady=10, fill=tk.X, padx=30)

        self.result_text = tk.Text(
            parent,
            height=15,
            width=60,
            wrap=tk.WORD,
            font=('Arial', 14),
            bg='white',
            padx=10,
            pady=10
        )
        self.result_text.pack(pady=10, fill=tk.BOTH, expand=True, padx=30)

    def add_heatmap_area(self, parent):
        self.heatmap_frame = tk.Frame(parent, bg='#f0f0f0')
        self.heatmap_frame.pack(pady=10, padx=30, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 8))
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.heatmap_frame)
        self.canvas_widget.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

    def open_dimension_window(self):
        dimensions = simpledialog.askstring(
            "Matrix Dimensions",
            "Enter dimensions (format: num_subjects,matrix_size)\nExample: 10,20",
            parent=self.root
        )
        if dimensions:
            try:
                num_subjects, matrix_size = map(int, dimensions.split(','))
                self.generate_simulated_data(num_subjects, matrix_size)
            except:
                messagebox.showerror("Error", "Invalid input format. Use: num_subjects,matrix_size")

    def generate_simulated_data(self, num_subjects, matrix_size):
        try:
            simulated_data = np.random.rand(num_subjects, matrix_size, matrix_size)
            self.analyzer = MatrixAnalyzer(simulated_data)
            self.result_text.insert(tk.END, f"Simulated data created: {num_subjects} subjects, {matrix_size}x{matrix_size} matrices\n")
            self.btn_show_ids.config(state=tk.NORMAL)
            self.btn_analyze.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")

    def import_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy Files", "*.npy"), ("CSV Files", "*.csv")])
        if file_path:
            try:
                if file_path.endswith('.npy'):
                    data = np.load(file_path)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    matrix_size = int(np.sqrt(df.shape[1]))
                    data = df.values.reshape((-1, matrix_size, matrix_size))
                if len(data.shape) != 3:
                    messagebox.showerror("Error", "Input data must be a 3D matrix.")
                    return
                self.analyzer = MatrixAnalyzer(data)
                self.result_text.insert(tk.END, f"Data imported successfully. {self.analyzer.num_subjects} subjects, {self.analyzer.data.shape[1]}x{self.analyzer.data.shape[2]} matrices\n")
                self.btn_show_ids.config(state=tk.NORMAL)
                self.btn_analyze.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import data: {str(e)}")

    def show_all_subject_ids(self):
        if self.analyzer:
            all_ids = ", ".join(map(str, self.analyzer.subject_ids))
            top = tk.Toplevel(self.root)
            top.title("All Subject IDs")
            text_widget = tk.Text(top, font=('Arial', 14), padx=20, pady=20)
            text_widget.insert(tk.END, all_ids)
            text_widget.pack()
            ok_button = tk.Button(top, text="OK", command=top.destroy, font=('Arial', 14))
            ok_button.pack(pady=10)

    def analyze_subject(self):
        if not self.analyzer:
            messagebox.showerror("Error", "Please import or generate data first")
            return

        try:
            subject_id = int(self.id_entry.get())
            matrix = self.analyzer.get_subject_matrix(subject_id)
            features = self.analyzer.calculate_features(matrix)
            if not features:
                return

            self.result_text.delete(1.0, tk.END)
            self.display_features(subject_id, features)
            self.display_heatmap(matrix)

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid subject ID: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")

    def display_features(self, subject_id, features):
        eigenvalues, singular_values, trace, determinant, clustering, centrality, diameter, path_length = features
        self.result_text.insert(tk.END, f"=== Subject ID: {subject_id} ===\n\n")
        self.result_text.insert(tk.END, "【Matrix Features】\n")
        self.result_text.insert(tk.END, f"Eigenvalues (first 5): {np.round(eigenvalues[:5], 2)}\n")
        self.result_text.insert(tk.END, f"Singular values (first 5): {np.round(singular_values[:5], 2)}\n")
        self.result_text.insert(tk.END, f"Trace: {trace:.2f} (Sum of diagonal elements)\n")
        self.result_text.insert(tk.END, f"Determinant: {determinant:.2e} (Matrix invertibility indicator)\n")
        self.result_text.insert(tk.END, f"Average Clustering: {clustering:.2f} (Local connectivity)\n")
        self.result_text.insert(tk.END, f"Average Degree Centrality: {centrality:.2f} (Global connectivity)\n")
        self.result_text.insert(tk.END, f"Network Diameter: {diameter:.2f} (Max path length)\n")
        self.result_text.insert(tk.END, f"Avg Shortest Path: {path_length:.2f} (Avg path length)\n")

    def display_heatmap(self, matrix):
        matrix_size = matrix.shape[0]
        fig_width = min(1000 / 100, matrix_size / 5) * 100  # 保持宽度不超过1000像素
        fig_height = fig_width * 0.8  # 保持比例
        self.fig.set_size_inches(fig_width / 100, fig_height / 100)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        im = ax.imshow(matrix, cmap='coolwarm', aspect='auto', interpolation='nearest')
        self.fig.colorbar(im, ax=ax, label='Connection Strength')
        ax.set_title(f"Connection Matrix (Size: {matrix_size}x{matrix_size})", fontsize=16)
        ax.set_xlabel("Nodes", fontsize=14)
        ax.set_ylabel("Nodes", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)

        self.canvas_widget.draw()

    def clear_data(self):
        self.analyzer = None
        self.btn_show_ids.config(state=tk.DISABLED)
        self.btn_analyze.config(state=tk.DISABLED)
        self.result_text.delete(1.0, tk.END)
        self.fig.clear()
        self.canvas_widget.draw()

    def on_close(self):
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = BrainNetworkApp(root)
    root.mainloop()