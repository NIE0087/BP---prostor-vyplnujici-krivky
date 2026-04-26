import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Hilbert3DVisualizer:
    def __init__(self, hilbert3d):
        self.hilbert = hilbert3d

    def _run_holder(self, H, I, r, eps, max_iter, n, whatFunc, true_min, ftol, stop_condition):
        return self.hilbert.Holder_algorithm(
            H=H,
            I=I,
            r=r,
            eps=eps,
            max_iter=max_iter,
            n=n,
            whatFunc=whatFunc,
            true_min=true_min,
            ftol=ftol,
            stop_condition=stop_condition,
        )

    def _create_axis_with_3d_fallback(self, figsize=(6, 6)):
        fig = plt.figure(figsize=figsize)
        try:
            ax = fig.add_subplot(projection="3d")
            return fig, ax, True
        except Exception:
            ax = fig.add_subplot()
            return fig, ax, False

    def _show_plotly_3d(self, pts, title, show_markers=False, show_labels=False, draw_grid=False, n=None):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="lines+markers" if show_markers else "lines",
                marker=dict(size=3, color="purple") if show_markers else None,
                line=dict(color="purple", width=3),
                name="curve",
            )
        )

        if draw_grid and n is not None:
            divs = 2 ** n
            grid = np.linspace(0, 1, divs + 1)
            for y in grid:
                for z in grid:
                    fig.add_trace(go.Scatter3d(x=[0, 1], y=[y, y], z=[z, z], mode="lines", line=dict(color="black", width=1), showlegend=False))
            for x in grid:
                for z in grid:
                    fig.add_trace(go.Scatter3d(x=[x, x], y=[0, 1], z=[z, z], mode="lines", line=dict(color="black", width=1), showlegend=False))
            for x in grid:
                for y in grid:
                    fig.add_trace(go.Scatter3d(x=[x, x], y=[y, y], z=[0, 1], mode="lines", line=dict(color="black", width=1), showlegend=False))

        if show_labels:
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="text",
                    text=[str(i + 1) for i in range(len(pts))],
                    textposition="top center",
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                zaxis=dict(range=[0, 1]),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()

    def compute_holder_table_values(
        self,
        n_values,
        whatFunc=0,
        H=-2,
        I=2,
        r=3,
        eps=1e-6,
        max_iter=1000,
        ftol=1e-6,
        true_min=None,
        true_point=None,
        print_table=True,
    ):
        if not n_values:
            raise ValueError("n_values nesmi byt prazdne")

        if whatFunc == 0:
            if true_min is None:
                true_min = 1.0
            if true_point is None:
                true_point = (0.3, 0.7, 0.0)
        elif true_min is None or true_point is None:
            raise ValueError("Pro whatFunc != 0 je potreba zadat true_min i true_point")

        y_star = np.asarray(true_point, dtype=float)
        if y_star.shape != (3,):
            raise ValueError("true_point musi byt trojice souradnic (x*, y*, z*)")

        rows = []
        for n in n_values:
            _, f_min, x_min, y_min, z_min, holder_iters = self._run_holder(
                H, I, r, eps, max_iter, n, whatFunc, true_min, ftol, stop_condition="ftol"
            )

            y_m = np.array([x_min, y_min, z_min], dtype=float)
            rows.append(
                {
                    "n": int(n),
                    "f_diff": float(f_min - true_min),
                    "y_x": float(x_min),
                    "y_y": float(y_min),
                    "y_z": float(z_min),
                    "distance": float(np.linalg.norm(y_m - y_star)),
                    "holder_iterations": int(holder_iters),
                }
            )

        df = pd.DataFrame(rows)
        if print_table:
            print(df.to_string(index=False))
        return df

    def holder_table_simple(
        self,
        n_values,
        whatFunc=0,
        H=-1,
        I=2,
        r=1.01,
        eps=1e-6,
        max_iter=1000,
        ftol=1e-6,
        true_min=1.0,
        true_point=(0.3, 0.7, 0.0),
    ):
        return self.compute_holder_table_values(
            n_values=n_values,
            whatFunc=whatFunc,
            H=H,
            I=I,
            r=r,
            eps=eps,
            max_iter=max_iter,
            ftol=ftol,
            true_min=true_min,
            true_point=true_point,
            print_table=True,
        )

    def compare_iteration_counts(
        self,
        n,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        whatFunc=0,
        true_min=None,
        ftol=1e-6,
        maxiter=1000,
        H=-2,
        I=2,
        r=3,
        eps=1e-6,
        max_iter_holder=1000,
        z_min=None,
        z_max=None,
    ):
        # Bounds are kept in the signature for backward compatibility.
        _ = (x_min, x_max, y_min, y_max, z_min, z_max)

        if true_min is None:
            raise ValueError("true_min musi byt zadano")

        results = {}

        try:
            f_min, x_mapped, y_mapped, z_mapped, generations, nfev = self.hilbert.differential_evolution_global(
                true_min=true_min,
                ftol=ftol,
                maxiter=maxiter,
                whatFunc=whatFunc,
            )
            results["differential_evolution"] = {
                "iterations": int(nfev),
                "generations": int(generations),
                "f_min": float(f_min),
                "x": float(x_mapped),
                "y": float(y_mapped),
                "z": float(z_mapped),
                "success": True,
            }
        except Exception as e:
            results["differential_evolution"] = {
                "iterations": None,
                "f_min": None,
                "success": False,
                "error": str(e),
            }

        try:
            _, f_min, x_mapped, y_mapped, z_mapped, holder_iterations = self._run_holder(
                H, I, r, eps, max_iter_holder, n, whatFunc, true_min, ftol, stop_condition="ftol"
            )

            results["holder"] = {
                "iterations": int(holder_iterations),
                "f_min": float(f_min),
                "x": float(x_mapped),
                "y": float(y_mapped),
                "z": float(z_mapped),
                "success": True,
            }
        except Exception as e:
            results["holder"] = {
                "iterations": None,
                "f_min": None,
                "success": False,
                "error": str(e),
            }

        return results

    def compare_iterations_by_curve_order(
        self,
        n_values,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        whatFunc=0,
        true_min=None,
        ftol=1e-6,
        maxiter=1000,
        H=-1,
        I=2,
        r=3,
        eps=1e-6,
        max_iter_holder=1000,
        z_min=None,
        z_max=None,
    ):
        all_results = {}
        for n in n_values:
            all_results[n] = self.compare_iteration_counts(
                n,
                x_min,
                x_max,
                y_min,
                y_max,
                whatFunc,
                true_min,
                ftol,
                maxiter,
                H,
                I,
                r,
                eps,
                max_iter_holder,
                z_min,
                z_max,
            )

        print("\n" + "=" * 70)
        print("SOUHRNNA TABULKA - POCET ITERACI PRO RUZNE RADY KRIVKY")
        print("=" * 70 + "\n")

        table_data = []
        for n in n_values:
            row = {"n": n}
            for method in ["differential_evolution", "holder"]:
                row[method] = all_results[n][method]["iterations"] if all_results[n][method]["success"] else "FAIL"
            table_data.append(row)

        df = pd.DataFrame(table_data)
        df.columns = ["n", "Diff. Evolution", "Holder"]
        print(df.to_string(index=False))
        return all_results, df

    def compare_algorithms(
        self,
        H,
        I,
        r,
        eps,
        max_iter,
        N_vals,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        whatFunc=0,
        true_min=None,
        stop_condition="eps",
        z_min=None,
        z_max=None,
    ):
        _ = (x_min, x_max, y_min, y_max, z_min, z_max)

        if true_min is None:
            raise ValueError("true_min musi byt zadano")

        results = []
        for n in N_vals:
            _, f_holder, _, _, _, _ = self._run_holder(
                H, I, r, eps, max_iter, n, whatFunc, true_min, eps, stop_condition=stop_condition
            )
            f_de, _, _, _, _, _ = self.hilbert.differential_evolution_global(
                true_min=true_min,
                ftol=eps,
                maxiter=max_iter,
                whatFunc=whatFunc,
            )
            results.append([n, f_holder, abs(f_holder - true_min), f_de, abs(f_de - true_min)])

        df = pd.DataFrame(
            results,
            columns=["Iterace n", "Hodnota Holder", "Rozdil Holder", "Hodnota DiffEvolution", "Rozdil DiffEvolution"],
        )
        print(df[["Iterace n", "Rozdil Holder", "Rozdil DiffEvolution"]])

        n_arr = df["Iterace n"].to_numpy()
        holder_arr = df["Rozdil Holder"].to_numpy()
        de_arr = df["Rozdil DiffEvolution"].to_numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(n_arr, holder_arr, "o-", label="Holder algoritmus")
        plt.plot(n_arr, de_arr, "s-", label="Differential Evolution")
        plt.xlabel("Iterace Hilbertovy krivky (n)")
        plt.ylabel("Rozdil od opravdoveho minima")
        plt.title("Porovnani optimalizacnich algoritmu")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.show()
        return df

    def compare_holder_variants(self, r, eps, max_iter, N_vals, whatFunc, true_min):
        variants = [
            ("HOLDER-CONST(-1), SELECT(1)", -1, 1),
            ("HOLDER-CONST(-1), SELECT(2)", -1, 2),
            ("HOLDER-CONST(-2), SELECT(1)", -2, 1),
            ("HOLDER-CONST(-2), SELECT(2)", -2, 2),
        ]

        results = {name: [] for name, _, _ in variants}
        n_values = []

        for n in N_vals:
            n_values.append(n)
            for name, H, I in variants:
                _, f_min, _, _, _, _ = self._run_holder(
                    H, I, r, eps, max_iter, n, whatFunc, true_min, eps, stop_condition="eps"
                )
                results[name].append(abs(f_min - true_min))

        plt.figure(figsize=(10, 6))
        colors = ["blue", "red", "green", "orange"]
        markers = ["o", "s", "^", "v"]

        for i, (name, _, _) in enumerate(variants):
            plt.plot(n_values, results[name], color=colors[i], marker=markers[i], label=name, linewidth=2, markersize=4)

        plt.xlabel("Iterace Hilbertovy krivky (n)")
        plt.ylabel("Rozdil od opravdoveho minima")
        plt.title("Porovnani variant Holderova algoritmu", fontsize=14)
        plt.yscale("log")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(fontsize=10, frameon=True, fancybox=True)
        plt.tight_layout()
        plt.show()

    def compare_H_approximations_iterations(self, H_exact, r, eps, max_iter, n_vals, whatFunc, true_min, ftol=None, I=2):
        eps_list = list(eps) if isinstance(eps, (list, tuple, np.ndarray)) else [eps]

        if ftol is None:
            ftol_list = eps_list.copy()
        elif isinstance(ftol, (list, tuple, np.ndarray)):
            ftol_list = list(ftol)
            if len(ftol_list) == 1 and len(eps_list) > 1:
                ftol_list = ftol_list * len(eps_list)
            elif len(ftol_list) != len(eps_list):
                raise ValueError("ftol musi mit stejny pocet hodnot jako eps (nebo jen jednu hodnotu)")
        else:
            ftol_list = [ftol] * len(eps_list)

        results = []
        for eps_val, ftol_val in zip(eps_list, ftol_list):
            for n in n_vals:
                _, f_min_exact, _, _, _, iter_exact = self._run_holder(
                    H_exact, I, r, eps_val, max_iter, n, whatFunc, true_min, ftol_val, stop_condition="eps"
                )
                _, f_min_h1, _, _, _, iter_h1 = self._run_holder(
                    -1, I, r, eps_val, max_iter, n, whatFunc, true_min, ftol_val, stop_condition="eps"
                )
                _, f_min_h2, _, _, _, iter_h2 = self._run_holder(
                    -2, I, r, eps_val, max_iter, n, whatFunc, true_min, ftol_val, stop_condition="eps"
                )

                results.append(
                    {
                        "eps": eps_val,
                        "ftol_used": ftol_val,
                        "n": n,
                        "H_exact_iter": int(iter_exact),
                        "H_exact_error": abs(float(f_min_exact) - float(true_min)),
                        "H=-1_iter": int(iter_h1),
                        "H=-1_error": abs(float(f_min_h1) - float(true_min)),
                        "H=-2_iter": int(iter_h2),
                        "H=-2_error": abs(float(f_min_h2) - float(true_min)),
                    }
                )

        df = pd.DataFrame(results)
        self._plot_H_comparison(df, eps_list)
        return df

    def _plot_H_comparison(self, df, eps_list):
        num_eps = len(eps_list)

        fig1, axes1 = plt.subplots(1, num_eps, figsize=(8 * num_eps, 6.5))
        if num_eps == 1:
            axes1 = [axes1]

        for idx, eps_val in enumerate(eps_list):
            ax = axes1[idx]
            df_tol = df[df["eps"] == eps_val]

            n_arr = df_tol["n"].to_numpy()
            ax.plot(n_arr, df_tol["H_exact_iter"].to_numpy(), "o-", label="H presne", linewidth=2, markersize=8)
            ax.plot(n_arr, df_tol["H=-1_iter"].to_numpy(), "s-", label="HOLDER-CONST(-1)", linewidth=2, markersize=8)
            ax.plot(n_arr, df_tol["H=-2_iter"].to_numpy(), "^-", label="HOLDER-CONST(-2)", linewidth=2, markersize=8)

            ax.set_xlabel("Řád Hilbertovy křivky (n)", fontsize=12)
            ax.set_ylabel("Počet iterací", fontsize=12)
            ax.set_title(f"Počet iterací pro eps={eps_val}", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        fig2, axes2 = plt.subplots(1, num_eps, figsize=(8 * num_eps, 6.5))
        if num_eps == 1:
            axes2 = [axes2]

        for idx, eps_val in enumerate(eps_list):
            ax = axes2[idx]
            df_tol = df[df["eps"] == eps_val]

            n_arr = df_tol["n"].to_numpy()
            ax.plot(n_arr, df_tol["H_exact_error"].to_numpy(), "o-", label="H presne", linewidth=2, markersize=8)
            ax.plot(n_arr, df_tol["H=-1_error"].to_numpy(), "s-", label="HOLDER-CONST(-1)", linewidth=2, markersize=8)
            ax.plot(n_arr, df_tol["H=-2_error"].to_numpy(), "^-", label="HOLDER-CONST(-2)", linewidth=2, markersize=8)

            ax.set_xlabel("Rad Hilbertovy krivky (n)", fontsize=12)
            ax.set_ylabel("Chyba |f_min - true_min|", fontsize=12)
            ax.set_title(f"Chyba pro eps={eps_val}", fontsize=13)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")
            ax.axhline(y=eps_val, color="r", linestyle="--", alpha=0.5)
            ax.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

    def compare_holder_variants_iterations(self, r, eps, max_iter, N_vals, whatFunc, true_min):
        variants = [
            ("HOLDER-CONST(-1), SELECT(1)", -1, 1),
            ("HOLDER-CONST(-1), SELECT(2)", -1, 2),
            ("HOLDER-CONST(-2), SELECT(1)", -2, 1),
            ("HOLDER-CONST(-2), SELECT(2)", -2, 2),
        ]

        results = {name: [] for name, _, _ in variants}
        n_values = []

        for n in N_vals:
            n_values.append(n)
            for name, H, I in variants:
                _, _, _, _, _, holder_iters = self._run_holder(
                    H, I, r, eps, max_iter, n, whatFunc, true_min, eps, stop_condition="eps"
                )
                results[name].append(int(holder_iters))

        plt.figure(figsize=(10, 6))
        colors = ["blue", "red", "green", "orange"]
        markers = ["o", "s", "^", "v"]

        for i, (name, _, _) in enumerate(variants):
            plt.plot(n_values, results[name], color=colors[i], marker=markers[i], label=name, linewidth=2, markersize=4)

        plt.xlabel("Iterace Hilbertovy křivky (n)")
        plt.ylabel("Počet iterací algoritmu")
        plt.title("Porovnání variant Hölderovského algoritmu podle počtu iterací", fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(fontsize=10, frameon=True, fancybox=True)
        plt.tight_layout()
        plt.show()

    def hyperparameter_tuning_r(self, r_values, H, I, eps, max_iter, N_vals, whatFunc, true_min, stop_condition="eps", ftol=None):
        results = {f"r={r}": [] for r in r_values}
        n_values = []
        ftol_used = eps if ftol is None else ftol

        for n in N_vals:
            n_values.append(n)
            for r_value in r_values:
                _, f_min, _, _, _, _ = self._run_holder(
                    H, I, r_value, eps, max_iter, n, whatFunc, true_min, ftol_used, stop_condition=stop_condition
                )
                results[f"r={r_value}"].append(abs(float(f_min) - float(true_min)))

        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(r_values)))
        markers = ["o", "s", "^", "v", "D", "P", "*", "X"]

        for i, r_value in enumerate(r_values):
            marker = markers[i % len(markers)]
            plt.plot(
                n_values,
                results[f"r={r_value}"],
                color=colors[i],
                marker=marker,
                linestyle="-",
                label=f"r={r_value}",
                linewidth=2,
                markersize=6,
                markerfacecolor="white",
                markeredgecolor=colors[i],
                markeredgewidth=2,
            )

        plt.xlabel("Iterace Hilbertovy křivky (n)", fontsize=12)
        plt.ylabel("Rozdíl od opravdového minima", fontsize=12)
        plt.title(f"Hyperparameter tuning r (H={H}, I={I})", fontsize=14)
        plt.yscale("log")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(fontsize=10, frameon=True, fancybox=True, ncol=2)
        plt.tight_layout()
        plt.show()

    def plot_hilbert_polygon(self, n):
        N = 8 ** n
        pts = np.zeros((N, 3))

        for k in range(N):
            t = k / N
            pts[k] = self.hilbert.hilbert_polygon_point(t, n)

        fig, ax, is_3d = self._create_axis_with_3d_fallback(figsize=(6, 6))
        if is_3d:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="purple", linewidth=0.5)
            ax.set_box_aspect([1, 1, 1])
        else:
            plt.close(fig)
            self._show_plotly_3d(pts, title="Hilbert polygon (Plotly 3D fallback)")
            return
        plt.show()

    def plot_hilbert_curve(self, n):
        samples = 8 ** n
        pts = np.zeros((samples, 3))

        for k in range(samples):
            t = k / samples
            q = self.hilbert.dec_to_octal(t)
            pts[k] = self.hilbert.ThreeD_Hilbert(q)

        fig, ax, is_3d = self._create_axis_with_3d_fallback(figsize=(6, 6))
        if is_3d:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="purple", linewidth=0.5)
            ax.set_box_aspect([1, 1, 1])
        else:
            plt.close(fig)
            self._show_plotly_3d(pts, title="Hilbert curve (Plotly 3D fallback)")
            return
        plt.show()

    def plot_mainstream_hilbert(self, n):
        samples = 8 ** n
        pts = np.zeros((samples, 3))

        for k in range(samples):
            t = k / samples
            q = self.hilbert.dec_to_octal(t)
            pts[k] = self.hilbert.calculate_mainstream_point(q, n)

        fig, ax, is_3d = self._create_axis_with_3d_fallback(figsize=(6, 6))
        if is_3d:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="purple", linewidth=0.5)
            ax.set_box_aspect([1, 1, 1])
        else:
            plt.close(fig)
            self._show_plotly_3d(pts, title="Mainstream Hilbert (Plotly 3D fallback)")
            return
        plt.show()

    def plot_mainstream_hilbert_cubes(self, n):
        samples = 8 ** n
        pts = np.zeros((samples, 3))
        for k in range(samples):
            t = k / samples
            q = self.hilbert.dec_to_octal(t)
            pts[k] = self.hilbert.calculate_mainstream_point(q, n)

        fig, ax, is_3d = self._create_axis_with_3d_fallback(figsize=(6, 6))
        ax.grid(False)
        if is_3d:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="purple", linewidth=0.5)
            ax.set_box_aspect([1, 1, 1])
        else:
            plt.close(fig)
            self._show_plotly_3d(
                pts,
                title="Mainstream Hilbert cubes (Plotly 3D fallback)",
                show_markers=True,
                show_labels=True,
                draw_grid=True,
                n=n,
            )
            return

        divs = 2 ** n
        grid = np.linspace(0, 1, divs + 1)

        for y in grid:
            for z in grid:
                ax.plot([0, 1], [y, y], [z, z], color="black", linewidth=0.9, alpha=0.5)

        for x in grid:
            for z in grid:
                ax.plot([x, x], [0, 1], [z, z], color="black", linewidth=0.9, alpha=0.5)

        for x in grid:
            for y in grid:
                ax.plot([x, x], [y, y], [0, 1], color="black", linewidth=0.9, alpha=0.5)

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="purple", s=15)

        for i, (x, y, z) in enumerate(pts):
            ax.text(x, y, z, str(i + 1), color="black", fontsize=8)

        ax.view_init(elev=20)
        plt.show()

    def plot_heat_source_holder_approximation(
        self,
        n,
        H=-2,
        I=2,
        r=3,
        eps=1e-6,
        max_iter=1000,
        stop_condition="eps",
        ftol=1e-6,
        heat_problem_config=None,
        cmap_source="viridis",
        cmap_temp="inferno",
    ):
        """Najde aproximaci zdroje pomoci Holdera (whatFunc=2) a vykresli porovnani poli."""
        if self.hilbert.heat_problem is None:
            config = {} if heat_problem_config is None else dict(heat_problem_config)
            self.hilbert.configure_heat_problem(**config)

        heat = self.hilbert.heat_problem

        # Pro stop_condition='eps' se true_min nepouziva, je zde jen kvuli API Holderu.
        t_min, f_min, x_est, y_est, z_est, holder_iters = self._run_holder(
            H,
            I,
            r,
            eps,
            max_iter,
            n,
            whatFunc=2,
            true_min=0.0,
            ftol=ftol,
            stop_condition=stop_condition,
        )

        A_est = float(self.hilbert.A_min + np.clip(z_est, 0.0, 1.0) * (self.hilbert.A_max - self.hilbert.A_min))

        x_true, y_true, A_true = map(float, heat.true_params)

        f_true = heat.source_term(x_true, y_true, A_true)
        f_est = heat.source_term(float(x_est), float(y_est), A_est)
        f_diff = f_est - f_true

        u_ref_clean = heat.u_ref_clean
        u_ref_noisy = heat.u_ref
        u_noise = u_ref_noisy - u_ref_clean
        u_est = heat.solve_forward((float(x_est), float(y_est), A_est))
        # Pouzij stejnou realizaci sumu i na odhadnutem poli pro prime vizualni porovnani.
        u_est_noisy = u_est + u_noise
        u_diff_noisy = u_est_noisy - u_ref_noisy
        u_diff_clean = u_est - u_ref_clean

        x_est_plot = float(x_est)
        y_est_plot = float(y_est)

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), constrained_layout=True)

        im00 = ax.imshow(
            u_ref_noisy.T,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap=cmap_temp,
            aspect="auto",
        )
  
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im00, ax=ax, label="u(x,y)")

        # Vyrazne oznaceni nalezene aproximace hvezdou.
        ax.scatter(
            x_est_plot,
            y_est_plot,
            marker="*",
            s=260,
            c="yellow",
            edgecolors="black",
            linewidths=1.3,
            zorder=10,
            label="Aproximace zdroje",
        )
        ax.scatter(
            x_true,
            y_true,
            marker="o",
            s=52,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
            label="Skutečný zdroj",
        )

        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        fig.suptitle(
            "Teplotní pole s nalezenou aproximací zdroje",
        )
        plt.show()

        return {
            "t_min": float(t_min),
            "cost_min": float(f_min),
            "holder_iterations": int(holder_iters),
            "estimated_params": (float(x_est), float(y_est), float(A_est)),
            "true_params": (x_true, y_true, A_true),
            "source_error_norm": float(np.linalg.norm(f_diff)),
            "temperature_error_norm_vs_noisy_ref": float(np.linalg.norm(u_diff_noisy)),
            "temperature_error_norm_vs_clean_ref": float(np.linalg.norm(u_diff_clean)),
        }

    def plot_heat_source_estimated_source_map(
        self,
        n,
        H=-2,
        I=2,
        r=3,
        eps=1e-6,
        max_iter=1000,
        stop_condition="eps",
        ftol=1e-6,
        heat_problem_config=None,
        cmap_source="viridis",
    ):
        """Vykresli samostatne mapu odhadnuteho zdroje nalezeneho Holderovym algoritmem."""
        if self.hilbert.heat_problem is None:
            config = {} if heat_problem_config is None else dict(heat_problem_config)
            self.hilbert.configure_heat_problem(**config)

        heat = self.hilbert.heat_problem

        t_min, f_min, x_est, y_est, z_est, holder_iters = self._run_holder(
            H,
            I,
            r,
            eps,
            max_iter,
            n,
            whatFunc=2,
            true_min=0.0,
            ftol=ftol,
            stop_condition=stop_condition,
        )

        A_est = float(self.hilbert.A_min + np.clip(z_est, 0.0, 1.0) * (self.hilbert.A_max - self.hilbert.A_min))
        x_true, y_true, A_true = map(float, heat.true_params)

        f_true = heat.source_term(x_true, y_true, A_true)
        f_est = heat.source_term(float(x_est), float(y_est), A_est)
        f_diff = f_est - f_true

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), constrained_layout=True)
        image = ax.imshow(
            f_est.T,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap=cmap_source,
            aspect="auto",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label="f(x,y,theta)")

        ax.scatter(
            float(x_est),
            float(y_est),
            marker="*",
            s=260,
            c="yellow",
            edgecolors="black",
            linewidths=1.3,
            zorder=10,
            label="Aproximace zdroje",
        )
        ax.scatter(
            x_true,
            y_true,
            marker="o",
            s=52,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
            label="Skutečný zdroj",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        fig.suptitle(
            "Aproximace zdroje"
        
        )
        plt.show()

        return {
            "t_min": float(t_min),
            "cost_min": float(f_min),
            "holder_iterations": int(holder_iters),
            "estimated_params": (float(x_est), float(y_est), float(A_est)),
            "true_params": (x_true, y_true, A_true),
            "source_error_norm": float(np.linalg.norm(f_diff)),
        }

    def plot_heat_source_holder_approximation_series(
        self,
        n_values,
        H=-2,
        I=2,
        r=3,
        eps=1e-6,
        max_iter=1000,
        stop_condition="eps",
        ftol=1e-6,
        heat_problem_config=None,
        cmap_source="viridis",
        cmap_temp="inferno",
    ):
        """Spusti vizualizaci Holder aproximace zdroje pro vice hodnot n."""
        if not n_values:
            raise ValueError("n_values nesmi byt prazdne")

        summaries = []
        for n in n_values:
            result = self.plot_heat_source_holder_approximation(
                n=n,
                H=H,
                I=I,
                r=r,
                eps=eps,
                max_iter=max_iter,
                stop_condition=stop_condition,
                ftol=ftol,
                heat_problem_config=heat_problem_config,
                cmap_source=cmap_source,
                cmap_temp=cmap_temp,
            )
            x_est, y_est, A_est = result["estimated_params"]
            summaries.append(
                {
                    "n": int(n),
                    "cost_min": float(result["cost_min"]),
                    "holder_iterations": int(result["holder_iterations"]),
                    "x_est": float(x_est),
                    "y_est": float(y_est),
                    "A_est": float(A_est),
                    "temperature_error_norm": float(result["temperature_error_norm_vs_noisy_ref"]),
                    "source_error_norm": float(result["source_error_norm"]),
                }
            )

        df = pd.DataFrame(summaries)
        print(df.to_string(index=False))
        return df




   






   






    