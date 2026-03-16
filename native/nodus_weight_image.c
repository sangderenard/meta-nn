/*
 * nodus_weight_image.c -- Weight-to-image renderer.
 *
 * Implements all three modes:
 *   parameter_groups    — square grid, 1 px per node group, nearest upscale
 *   architectural_tall  — layers as vertical columns, 1 px per unit
 *   architectural_wide  — transposed: layers as horizontal rows
 *
 * The colour palette and statistics-to-colour mapping exactly mirror the
 * Python implementation in pipeline/weight_map.py.
 */

#include "nodus_weight_image.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/*  Colour palette  (matches _WEIGHT_MAP_* in weight_map.py)          */
/* ------------------------------------------------------------------ */

static const float BG_R  = 14.0f,  BG_G  = 16.0f,  BG_B  = 20.0f;
static const float WM_R  = 246.0f, WM_G  = 127.0f, WM_B  = 33.0f;   /* warm  */
static const float CL_R  = 44.0f,  CL_G  = 140.0f, CL_B  = 255.0f;  /* cool  */
static const float NE_R  = 120.0f, NE_G  = 196.0f, NE_B  = 130.0f;  /* neutral */
static const float DR_R  = 255.0f, DR_G  = 52.0f,  DR_B  = 52.0f;   /* drift */

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float maxf(float a, float b) { return a > b ? a : b; }

static int maxi(int a, int b) { return a > b ? a : b; }
static int mini(int a, int b) { return a < b ? a : b; }

/* Robust scale: 95th percentile of absolute values. */
static float robust_scale(const float *values, int count) {
    if (count <= 0) return 1.0f;

    /* Collect abs values into temp array on the stack for small counts,
       heap for large. */
    float stack_buf[256];
    float *buf = (count <= 256) ? stack_buf : (float *)malloc((size_t)count * sizeof(float));
    if (!buf) return 1.0f;

    for (int i = 0; i < count; i++)
        buf[i] = fabsf(values[i]);

    /* Simple insertion sort — layers/units are typically small. */
    for (int i = 1; i < count; i++) {
        float key = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j] > key) {
            buf[j + 1] = buf[j];
            j--;
        }
        buf[j + 1] = key;
    }

    int idx95 = (int)((float)(count - 1) * 0.95f);
    if (idx95 < 0) idx95 = 0;
    if (idx95 >= count) idx95 = count - 1;
    float scale = buf[idx95];

    if (!(scale > 1e-12f))
        scale = buf[count - 1]; /* max */
    if (!(scale > 1e-12f))
        scale = 1.0f;

    if (buf != stack_buf) free(buf);
    return scale;
}

/* ------------------------------------------------------------------ */
/*  _node_color — identical logic to Python's _node_color()           */
/* ------------------------------------------------------------------ */

static void node_color(
        float s_mean,  float s_mean_abs, float s_rms, float s_diff_mean_abs,
        float max_abs_mean, float max_energy, float max_diff,
        uint8_t *out_r, uint8_t *out_g, uint8_t *out_b)
{
    float signed_v = s_mean / maxf(1e-12f, max_abs_mean);
    float energy   = s_mean_abs / maxf(1e-12f, max_energy);
    float drift    = s_diff_mean_abs / maxf(1e-12f, max_diff);

    signed_v = clampf(signed_v, -1.0f, 1.0f);
    energy   = clampf(energy,    0.0f, 1.0f);
    drift    = clampf(drift,     0.0f, 1.0f);

    float warm    = maxf(0.0f,  signed_v);
    float cool    = maxf(0.0f, -signed_v);
    float neutral = maxf(0.0f, 1.0f - fabsf(signed_v));

    float base_r = warm * WM_R + cool * CL_R + neutral * NE_R;
    float base_g = warm * WM_G + cool * CL_G + neutral * NE_G;
    float base_b = warm * WM_B + cool * CL_B + neutral * NE_B;

    float intensity = 0.18f + 0.82f * energy;
    float cr = BG_R * (1.0f - intensity) + base_r * intensity;
    float cg = BG_G * (1.0f - intensity) + base_g * intensity;
    float cb = BG_B * (1.0f - intensity) + base_b * intensity;

    if (drift > 0.0f) {
        float blend = 0.18f + 0.34f * drift;
        cr = cr * (1.0f - blend) + DR_R * blend;
        cg = cg * (1.0f - blend) + DR_G * blend;
        cb = cb * (1.0f - blend) + DR_B * blend;
    }

    *out_r = (uint8_t)clampf(cr, 0.0f, 255.0f);
    *out_g = (uint8_t)clampf(cg, 0.0f, 255.0f);
    *out_b = (uint8_t)clampf(cb, 0.0f, 255.0f);
}

/* ------------------------------------------------------------------ */
/*  Fill canvas with BG colour                                        */
/* ------------------------------------------------------------------ */

static void fill_bg(uint8_t *rgb, int w, int h) {
    uint8_t br = (uint8_t)BG_R, bg = (uint8_t)BG_G, bb = (uint8_t)BG_B;
    int n = w * h;
    for (int i = 0; i < n; i++) {
        rgb[i * 3 + 0] = br;
        rgb[i * 3 + 1] = bg;
        rgb[i * 3 + 2] = bb;
    }
}

static void fill_black(uint8_t *rgb, int w, int h) {
    memset(rgb, 0, (size_t)w * (size_t)h * 3);
}

/* Set a single pixel (bounds-checked). */
static void set_pixel(uint8_t *rgb, int stride_w, int total_h,
                       int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= stride_w || y < 0 || y >= total_h) return;
    int off = (y * stride_w + x) * 3;
    rgb[off + 0] = r;
    rgb[off + 1] = g;
    rgb[off + 2] = b;
}

/* ------------------------------------------------------------------ */
/*  Nearest-neighbour upscale of raw grid into a larger buffer.       */
/*  Caller must ensure dst has room for (src_w*sx) * (src_h*sy) * 3.  */
/* ------------------------------------------------------------------ */

static void nn_upscale(const uint8_t *src, int src_w, int src_h,
                       uint8_t *dst, int sx, int sy)
{
    int dst_w = src_w * sx;
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            int si = (y * src_w + x) * 3;
            uint8_t r = src[si], g = src[si + 1], b = src[si + 2];
            for (int dy = 0; dy < sy; dy++) {
                for (int dx = 0; dx < sx; dx++) {
                    int di = ((y * sy + dy) * dst_w + (x * sx + dx)) * 3;
                    dst[di]     = r;
                    dst[di + 1] = g;
                    dst[di + 2] = b;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  parameter_groups mode                                             */
/* ------------------------------------------------------------------ */

static int render_parameter_groups(
        const NodusWeightNodeGroup *nodes, int32_t num_nodes,
        int32_t target_w, int32_t target_h,
        uint8_t *out_rgb, int32_t *out_w, int32_t *out_h)
{
    int side_px = maxi(8, maxi(target_w, target_h));
    int nn = maxi(1, num_nodes);
    int grid_side = (int)ceilf(sqrtf((float)nn));
    if (grid_side < 1) grid_side = 1;

    /* Build tiny grid at 1 px per node. */
    int grid_bytes = grid_side * grid_side * 3;
    uint8_t *grid = (uint8_t *)malloc((size_t)grid_bytes);
    if (!grid) return -1;
    fill_bg(grid, grid_side, grid_side);

    if (num_nodes > 0) {
        /* Compute normalization scales. */
        float *abs_means  = (float *)malloc((size_t)num_nodes * sizeof(float));
        float *energies   = (float *)malloc((size_t)num_nodes * sizeof(float));
        float *diffs      = (float *)malloc((size_t)num_nodes * sizeof(float));
        if (!abs_means || !energies || !diffs) {
            free(abs_means); free(energies); free(diffs); free(grid);
            return -1;
        }
        for (int i = 0; i < num_nodes; i++) {
            float am = fabsf(nodes[i].mean);
            float ma = nodes[i].mean_abs;
            abs_means[i] = maxf(am, ma);
            energies[i]  = ma;
            diffs[i]     = nodes[i].diff_mean_abs;
        }
        float max_abs_mean = robust_scale(abs_means, num_nodes);
        float max_energy   = robust_scale(energies,  num_nodes);
        float max_diff     = robust_scale(diffs,     num_nodes);
        if (max_abs_mean < 1e-12f) max_abs_mean = 1.0f;
        if (max_energy   < 1e-12f) max_energy   = 1.0f;
        if (max_diff     < 1e-12f) max_diff     = 1.0f;

        free(abs_means); free(energies); free(diffs);

        for (int i = 0; i < num_nodes; i++) {
            int gy = i / grid_side;
            int gx = i % grid_side;
            uint8_t r, g, b;
            node_color(nodes[i].mean, nodes[i].mean_abs,
                       nodes[i].rms,  nodes[i].diff_mean_abs,
                       max_abs_mean, max_energy, max_diff,
                       &r, &g, &b);
            set_pixel(grid, grid_side, grid_side, gx, gy, r, g, b);
        }
    }

    /* Nearest-neighbour upscale to side_px x side_px. */
    int sx = maxi(1, side_px / grid_side);
    int sy = sx;
    int scaled_w = grid_side * sx;
    int scaled_h = grid_side * sy;

    uint8_t *scaled = (uint8_t *)malloc((size_t)scaled_w * (size_t)scaled_h * 3);
    if (!scaled) { free(grid); return -1; }
    nn_upscale(grid, grid_side, grid_side, scaled, sx, sy);
    free(grid);

    /* Centre in side_px x side_px canvas with BG padding. */
    fill_bg(out_rgb, side_px, side_px);
    int x0 = (side_px - scaled_w) / 2;
    int y0 = (side_px - scaled_h) / 2;
    for (int y = 0; y < scaled_h && (y + y0) < side_px; y++) {
        for (int x = 0; x < scaled_w && (x + x0) < side_px; x++) {
            int si = (y * scaled_w + x) * 3;
            int di = ((y + y0) * side_px + (x + x0)) * 3;
            out_rgb[di]     = scaled[si];
            out_rgb[di + 1] = scaled[si + 1];
            out_rgb[di + 2] = scaled[si + 2];
        }
    }
    free(scaled);

    *out_w = side_px;
    *out_h = side_px;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  architectural mode (tall or wide)                                 */
/* ------------------------------------------------------------------ */

/* Internal layout entry for one layer. */
typedef struct LayerLayout {
    int unit_count;
    int cols;       /* columns occupied (tall) or rows (wide) */
    int x0;        /* raw grid column start */
    int x1;        /* raw grid column end */
    float max_abs_mean;
    float max_energy;
    float max_diff;
} LayerLayout;

static int render_architectural(
        const NodusWeightLayer *layers, int32_t num_layers,
        int32_t target_w, int32_t target_h,
        int transpose,  /* 0 = tall, 1 = wide */
        uint8_t *out_rgb, int32_t *out_w, int32_t *out_h)
{
    if (num_layers <= 0 || !layers) return -1;

    int footer_h = 18;

    /* Find max unit count across all layers. */
    int max_unit_count = 1;
    for (int i = 0; i < num_layers; i++) {
        if (layers[i].unit_count > max_unit_count)
            max_unit_count = layers[i].unit_count;
    }
    /* Force odd for symmetric centering. */
    int raw_h = (max_unit_count % 2 == 1) ? max_unit_count : max_unit_count + 1;
    int row_capacity = raw_h;

    /* Per-layer layout. */
    LayerLayout *ll = (LayerLayout *)malloc((size_t)num_layers * sizeof(LayerLayout));
    if (!ll) return -1;

    /* Temp buffers for robust_scale per layer. */
    float *tmp_vals = (float *)malloc((size_t)(max_unit_count > 0 ? max_unit_count : 1) * sizeof(float));
    if (!tmp_vals) { free(ll); return -1; }

    int raw_total_w = 0;
    for (int li = 0; li < num_layers; li++) {
        int uc = maxi(1, layers[li].unit_count);
        int cols = maxi(1, (int)ceilf((float)uc / (float)row_capacity));
        ll[li].unit_count = layers[li].unit_count;
        ll[li].cols = cols;
        ll[li].x0 = raw_total_w;
        ll[li].x1 = raw_total_w + cols;
        raw_total_w += cols;

        /* Compute per-layer normalization. */
        int n = layers[li].unit_count;
        const NodusWeightUnit *units = layers[li].units;

        for (int u = 0; u < n; u++) tmp_vals[u] = fabsf(units[u].mean);
        ll[li].max_abs_mean = robust_scale(tmp_vals, n);
        if (ll[li].max_abs_mean < 1e-12f) ll[li].max_abs_mean = 1.0f;

        for (int u = 0; u < n; u++) tmp_vals[u] = units[u].mean_abs;
        ll[li].max_energy = robust_scale(tmp_vals, n);
        if (ll[li].max_energy < 1e-12f) ll[li].max_energy = 1.0f;

        for (int u = 0; u < n; u++) tmp_vals[u] = units[u].diff_mean_abs;
        ll[li].max_diff = robust_scale(tmp_vals, n);
        if (ll[li].max_diff < 1e-12f) ll[li].max_diff = 1.0f;
    }
    free(tmp_vals);

    int raw_w = maxi(1, raw_total_w);

    /* Build raw grid (1 px per unit). */
    uint8_t *raw = (uint8_t *)malloc((size_t)raw_h * (size_t)raw_w * 3);
    if (!raw) { free(ll); return -1; }
    fill_bg(raw, raw_w, raw_h);

    for (int li = 0; li < num_layers; li++) {
        int n = layers[li].unit_count;
        const NodusWeightUnit *units = layers[li].units;
        for (int idx = 0; idx < n; idx++) {
            int col = idx / row_capacity;
            int pos_in_col = idx % row_capacity;
            int units_in_col = mini(row_capacity, maxi(1, n - col * row_capacity));
            int y_offset = (row_capacity - units_in_col) / 2;
            int gx = ll[li].x0 + col;
            int gy = y_offset + pos_in_col;
            if (gy >= 0 && gy < raw_h && gx >= 0 && gx < raw_w) {
                uint8_t r, g, b;
                node_color(units[idx].mean, units[idx].mean_abs,
                           units[idx].rms,  units[idx].diff_mean_abs,
                           ll[li].max_abs_mean, ll[li].max_energy,
                           ll[li].max_diff,
                           &r, &g, &b);
                set_pixel(raw, raw_w, raw_h, gx, gy, r, g, b);
            }
        }
    }

    /* For wide mode, transpose the raw grid. */
    uint8_t *oriented;
    int ori_w, ori_h;
    if (transpose) {
        ori_w = raw_h;
        ori_h = raw_w;
        oriented = (uint8_t *)malloc((size_t)ori_w * (size_t)ori_h * 3);
        if (!oriented) { free(raw); free(ll); return -1; }
        for (int y = 0; y < raw_h; y++) {
            for (int x = 0; x < raw_w; x++) {
                int si = (y * raw_w + x) * 3;
                int di = (x * ori_w + y) * 3;
                oriented[di]     = raw[si];
                oriented[di + 1] = raw[si + 1];
                oriented[di + 2] = raw[si + 2];
            }
        }
        free(raw);
    } else {
        oriented = raw;
        ori_w = raw_w;
        ori_h = raw_h;
    }

    /* Effective target: never smaller than oriented grid. */
    int eff_w = maxi(target_w, ori_w);
    int eff_h = maxi(target_h, ori_h + footer_h);
    int data_h = eff_h - footer_h;

    /* Integer scale: largest factor that fits within target. */
    int scale = maxi(1, mini(data_h / ori_h, eff_w / ori_w));
    int scaled_w = ori_w * scale;
    int scaled_h = ori_h * scale;

    /* Upscale. */
    uint8_t *scaled;
    if (scale > 1) {
        scaled = (uint8_t *)malloc((size_t)scaled_w * (size_t)scaled_h * 3);
        if (!scaled) { free(oriented); free(ll); return -1; }
        nn_upscale(oriented, ori_w, ori_h, scaled, scale, scale);
        free(oriented);
    } else {
        scaled = oriented;
    }

    /* Black canvas of exactly (eff_h x eff_w). */
    fill_black(out_rgb, eff_w, eff_h);

    /* Centre the scaled grid in the data area. */
    int y0 = (data_h - scaled_h) / 2;
    int x0 = (eff_w - scaled_w) / 2;
    for (int y = 0; y < scaled_h; y++) {
        int dy = y + y0;
        if (dy < 0 || dy >= data_h) continue;
        for (int x = 0; x < scaled_w; x++) {
            int dx = x + x0;
            if (dx < 0 || dx >= eff_w) continue;
            int si = (y * scaled_w + x) * 3;
            int di = (dy * eff_w + dx) * 3;
            out_rgb[di]     = scaled[si];
            out_rgb[di + 1] = scaled[si + 1];
            out_rgb[di + 2] = scaled[si + 2];
        }
    }
    free(scaled);

    /* Footer background: dark bar at the bottom. */
    for (int y = data_h; y < eff_h; y++) {
        for (int x = 0; x < eff_w; x++) {
            set_pixel(out_rgb, eff_w, eff_h, x, y, 8, 10, 14);
        }
    }
    /* Separator line. */
    for (int x = 0; x < eff_w; x++) {
        set_pixel(out_rgb, eff_w, eff_h, x, data_h, 52, 58, 68);
    }
    /* Tick marks at each layer center. */
    for (int li = 0; li < num_layers; li++) {
        int lx0, lx1, cx;
        if (transpose) {
            /* In wide mode the layers run vertically after transpose,
               but footer ticks still index by the pre-transpose column
               range mapped through scale + centering. */
            lx0 = x0 + ll[li].x0 * scale;
            lx1 = x0 + ll[li].x1 * scale;
        } else {
            lx0 = x0 + ll[li].x0 * scale;
            lx1 = x0 + ll[li].x1 * scale;
        }
        if (lx1 <= lx0) lx1 = lx0 + 1;
        cx = (lx0 + lx1) / 2;
        for (int y = data_h; y < eff_h; y++) {
            set_pixel(out_rgb, eff_w, eff_h, cx, y, 76, 84, 96);
        }
    }

    free(ll);

    *out_w = eff_w;
    *out_h = eff_h;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Public entry point                                                */
/* ------------------------------------------------------------------ */

NODUS_API int nodus_weight_image_render(
        int                         mode,
        const NodusWeightLayer     *layers,
        int32_t                     num_layers,
        const NodusWeightNodeGroup *nodes,
        int32_t                     num_nodes,
        int32_t                     target_w,
        int32_t                     target_h,
        uint8_t                    *out_rgb,
        int32_t                    *out_w,
        int32_t                    *out_h)
{
    if (!out_rgb || !out_w || !out_h) return -1;
    target_w = maxi(8, target_w);
    target_h = maxi(8, target_h);

    switch (mode) {
    case NODUS_WEIGHT_MODE_PARAMETER_GROUPS:
        if (!nodes || num_nodes <= 0) return -1;
        return render_parameter_groups(nodes, num_nodes,
                                       target_w, target_h,
                                       out_rgb, out_w, out_h);

    case NODUS_WEIGHT_MODE_ARCHITECTURAL_TALL:
        if (!layers || num_layers <= 0) return -1;
        return render_architectural(layers, num_layers,
                                    target_w, target_h,
                                    0, out_rgb, out_w, out_h);

    case NODUS_WEIGHT_MODE_ARCHITECTURAL_WIDE:
        if (!layers || num_layers <= 0) return -1;
        return render_architectural(layers, num_layers,
                                    target_w, target_h,
                                    1, out_rgb, out_w, out_h);

    default:
        return -1;
    }
}

/* ================================================================== */
/*  State-dict intake — raw tensors in, image out, all work in C      */
/* ================================================================== */

/* Extract the node key from a parameter name by dropping the last
   dotted component.  E.g. "block.weight" -> "block",
   "features.0.conv.bias" -> "features.0.conv".
   If there's no dot, returns "<root>".  Writes into buf. */
static void param_node_key(const char *name, char *buf, int buf_len) {
    if (!name || !name[0] || buf_len <= 0) {
        if (buf_len > 0) { buf[0] = '\0'; }
        return;
    }
    int len = (int)strlen(name);
    /* Find last dot. */
    int last_dot = -1;
    for (int i = len - 1; i >= 0; i--) {
        if (name[i] == '.') { last_dot = i; break; }
    }
    if (last_dot <= 0) {
        /* No dot or dot at position 0 → root. */
        int n = 6 < (buf_len - 1) ? 6 : (buf_len - 1);
        memcpy(buf, "<root>", (size_t)n);
        buf[n] = '\0';
        return;
    }
    int copy = last_dot < (buf_len - 1) ? last_dot : (buf_len - 1);
    memcpy(buf, name, (size_t)copy);
    buf[copy] = '\0';
}

/* Internal node-group accumulator for grouping parameters. */
typedef struct NodeAccum {
    char     name[256];
    int      param_indices[4096];   /* indices into the param arrays */
    int      num_params;
    /* For architectural mode: unit_count = mode dim-0 count. */
    int      unit_count;
} NodeAccum;

/* Choose the most common shape0 among a set of parameters as the
   unit_count for a node group (matches Python's _choose_unit_count). */
static int choose_unit_count(const int32_t *shape0_vals, int count) {
    if (count <= 0) return 1;
    /* Frequency count via simple scan — usually < 10 params per node. */
    int best_val = shape0_vals[0];
    int best_freq = 0;
    for (int i = 0; i < count; i++) {
        int val = shape0_vals[i];
        int freq = 0;
        for (int j = 0; j < count; j++) {
            if (shape0_vals[j] == val) freq++;
        }
        if (freq > best_freq || (freq == best_freq && val > best_val)) {
            best_freq = freq;
            best_val = val;
        }
    }
    return best_val > 0 ? best_val : 1;
}

NODUS_API int nodus_weight_image_from_state_dict(
        const char *const  *param_names,
        const float *const *param_data,
        const int32_t      *param_numel,
        const int32_t      *param_shape0,
        int32_t             num_params,
        const float *const *ref_data,
        const int32_t      *ref_numel,
        int                 mode,
        int32_t             target_w,
        int32_t             target_h,
        uint8_t           **out_rgb,
        int32_t            *out_w,
        int32_t            *out_h)
{
    if (!param_names || !param_data || !param_numel || !param_shape0
        || num_params <= 0 || !out_rgb || !out_w || !out_h)
        return -1;

    target_w = maxi(8, target_w);
    target_h = maxi(8, target_h);

    /* ---- Group parameters by node key ---- */
    NodeAccum *nodes_acc = (NodeAccum *)calloc((size_t)num_params, sizeof(NodeAccum));
    if (!nodes_acc) return -1;
    int num_groups = 0;

    for (int pi = 0; pi < num_params; pi++) {
        char key[256];
        param_node_key(param_names[pi], key, 256);

        /* Find existing group. */
        int gi = -1;
        for (int g = 0; g < num_groups; g++) {
            if (strcmp(nodes_acc[g].name, key) == 0) { gi = g; break; }
        }
        if (gi < 0) {
            gi = num_groups++;
            memcpy(nodes_acc[gi].name, key, 256);
            nodes_acc[gi].num_params = 0;
        }
        if (nodes_acc[gi].num_params < 4096) {
            nodes_acc[gi].param_indices[nodes_acc[gi].num_params++] = pi;
        }
    }

    /* ---- Compute unit_count per group ---- */
    int32_t *shape0_buf = (int32_t *)malloc((size_t)num_params * sizeof(int32_t));
    if (!shape0_buf) { free(nodes_acc); return -1; }

    for (int g = 0; g < num_groups; g++) {
        int np = nodes_acc[g].num_params;
        for (int j = 0; j < np; j++) {
            shape0_buf[j] = param_shape0[nodes_acc[g].param_indices[j]];
        }
        nodes_acc[g].unit_count = choose_unit_count(shape0_buf, np);
    }
    free(shape0_buf);

    /* ---- PARAMETER_GROUPS mode: aggregate per node group ---- */
    if (mode == NODUS_WEIGHT_MODE_PARAMETER_GROUPS) {
        NodusWeightNodeGroup *ng = (NodusWeightNodeGroup *)calloc(
            (size_t)num_groups, sizeof(NodusWeightNodeGroup));
        if (!ng) { free(nodes_acc); return -1; }

        for (int g = 0; g < num_groups; g++) {
            double sum = 0.0, abs_sum = 0.0, sq_sum = 0.0, diff_abs_sum = 0.0;
            int64_t total_elems = 0;

            for (int j = 0; j < nodes_acc[g].num_params; j++) {
                int pi = nodes_acc[g].param_indices[j];
                int32_t n = param_numel[pi];
                const float *data = param_data[pi];
                for (int32_t k = 0; k < n; k++) {
                    float v = data[k];
                    sum += (double)v;
                    abs_sum += (double)fabsf(v);
                    sq_sum += (double)(v * v);
                }
                total_elems += n;

                if (ref_data && ref_data[pi] && ref_numel && ref_numel[pi] == n) {
                    const float *rd = ref_data[pi];
                    for (int32_t k = 0; k < n; k++) {
                        diff_abs_sum += (double)fabsf(data[k] - rd[k]);
                    }
                }
            }

            double count = (double)(total_elems > 0 ? total_elems : 1);
            ng[g].mean          = (float)(sum / count);
            ng[g].mean_abs      = (float)(abs_sum / count);
            ng[g].rms           = (float)sqrt(sq_sum / count);
            ng[g].diff_mean_abs = (float)(diff_abs_sum / count);
        }

        /* Allocate output — C owns this memory. */
        int32_t side = maxi(target_w, target_h);
        side = maxi(8, side);
        uint8_t *rgb = (uint8_t *)malloc((size_t)side * (size_t)side * 3);
        if (!rgb) { free(ng); free(nodes_acc); return -1; }

        int rc = render_parameter_groups(ng, num_groups,
                                         target_w, target_h,
                                         rgb, out_w, out_h);
        free(ng);
        free(nodes_acc);
        if (rc != 0) { free(rgb); return -1; }
        *out_rgb = rgb;
        return 0;
    }

    /* ---- ARCHITECTURAL modes: build per-layer per-unit stats ---- */
    NodusWeightLayer *layers = (NodusWeightLayer *)calloc(
        (size_t)num_groups, sizeof(NodusWeightLayer));
    if (!layers) { free(nodes_acc); return -1; }

    for (int g = 0; g < num_groups; g++) {
        int uc = nodes_acc[g].unit_count;
        layers[g].unit_count = uc;
        layers[g].units = (NodusWeightUnit *)calloc((size_t)uc, sizeof(NodusWeightUnit));
        if (!layers[g].units) {
            for (int k = 0; k < g; k++) free(layers[k].units);
            free(layers); free(nodes_acc);
            return -1;
        }

        /* Accumulators per unit. */
        double *u_sum     = (double *)calloc((size_t)uc, sizeof(double));
        double *u_abs_sum = (double *)calloc((size_t)uc, sizeof(double));
        double *u_sq_sum  = (double *)calloc((size_t)uc, sizeof(double));
        double *u_diff    = (double *)calloc((size_t)uc, sizeof(double));
        double *u_count   = (double *)calloc((size_t)uc, sizeof(double));
        if (!u_sum || !u_abs_sum || !u_sq_sum || !u_diff || !u_count) {
            free(u_sum); free(u_abs_sum); free(u_sq_sum); free(u_diff); free(u_count);
            for (int k = 0; k <= g; k++) free(layers[k].units);
            free(layers); free(nodes_acc);
            return -1;
        }

        for (int j = 0; j < nodes_acc[g].num_params; j++) {
            int pi = nodes_acc[g].param_indices[j];
            int32_t n = param_numel[pi];
            int32_t s0 = param_shape0[pi];
            const float *data = param_data[pi];
            const float *rd = NULL;
            int has_ref = 0;
            if (ref_data && ref_data[pi] && ref_numel && ref_numel[pi] == n) {
                rd = ref_data[pi];
                has_ref = 1;
            }

            /* Can we decompose this tensor into uc rows? */
            if (s0 == uc && n > 0) {
                /* Tensor shape is [uc, ...], each row has n/uc elements. */
                int32_t row_len = n / uc;
                for (int u = 0; u < uc; u++) {
                    const float *row = data + (int64_t)u * (int64_t)row_len;
                    for (int32_t k = 0; k < row_len; k++) {
                        float v = row[k];
                        u_sum[u]     += (double)v;
                        u_abs_sum[u] += (double)fabsf(v);
                        u_sq_sum[u]  += (double)(v * v);
                    }
                    u_count[u] += (double)row_len;

                    if (has_ref) {
                        const float *rrow = rd + (int64_t)u * (int64_t)row_len;
                        for (int32_t k = 0; k < row_len; k++) {
                            u_diff[u] += (double)fabsf(row[k] - rrow[k]);
                        }
                    }
                }
            } else if (n == uc) {
                /* 1D tensor of exactly uc elements — one per unit. */
                for (int u = 0; u < uc; u++) {
                    float v = data[u];
                    u_sum[u]     += (double)v;
                    u_abs_sum[u] += (double)fabsf(v);
                    u_sq_sum[u]  += (double)(v * v);
                    u_count[u]   += 1.0;
                    if (has_ref) {
                        u_diff[u] += (double)fabsf(v - rd[u]);
                    }
                }
            }
            /* else: skip tensors that don't decompose into uc rows
               (matches Python's _unit_rows_for_tensor returning None) */
        }

        for (int u = 0; u < uc; u++) {
            double c = u_count[u] > 0.0 ? u_count[u] : 1.0;
            layers[g].units[u].mean          = (float)(u_sum[u] / c);
            layers[g].units[u].mean_abs      = (float)(u_abs_sum[u] / c);
            layers[g].units[u].rms           = (float)sqrt(u_sq_sum[u] / c);
            layers[g].units[u].diff_mean_abs = (float)(u_diff[u] / c);
        }

        free(u_sum); free(u_abs_sum); free(u_sq_sum); free(u_diff); free(u_count);
    }

    free(nodes_acc);

    /* Compute max possible image size so we can heap-allocate. */
    int max_units = 1;
    int total_cols = 0;
    for (int g = 0; g < num_groups; g++) {
        if (layers[g].unit_count > max_units)
            max_units = layers[g].unit_count;
        int uc = maxi(1, layers[g].unit_count);
        int raw_h_est = (max_units % 2 == 1) ? max_units : max_units + 1;
        int cols = maxi(1, (int)ceilf((float)uc / (float)raw_h_est));
        total_cols += cols;
    }
    int footer_h = 18;
    int raw_h = (max_units % 2 == 1) ? max_units : max_units + 1;
    int eff_w = maxi(target_w, total_cols);
    int eff_h = maxi(target_h, raw_h + footer_h);
    /* For wide mode, transpose dimensions. */
    if (mode == NODUS_WEIGHT_MODE_ARCHITECTURAL_WIDE) {
        int tw = maxi(target_w, raw_h);
        int th = maxi(target_h, total_cols + footer_h);
        eff_w = maxi(eff_w, tw);
        eff_h = maxi(eff_h, th);
    }
    /* Conservative upper bound: scale can only increase dimensions. */
    int alloc_w = maxi(eff_w, target_w);
    int alloc_h = maxi(eff_h, target_h);

    uint8_t *rgb = (uint8_t *)malloc((size_t)alloc_w * (size_t)alloc_h * 3);
    if (!rgb) {
        for (int g = 0; g < num_groups; g++) free(layers[g].units);
        free(layers);
        return -1;
    }

    int transpose = (mode == NODUS_WEIGHT_MODE_ARCHITECTURAL_WIDE) ? 1 : 0;
    int rc = render_architectural(layers, num_groups,
                                  target_w, target_h,
                                  transpose, rgb, out_w, out_h);

    for (int g = 0; g < num_groups; g++) free(layers[g].units);
    free(layers);

    if (rc != 0) { free(rgb); return -1; }

    /* If the actual image is smaller than what we allocated, realloc to
       exact size so the caller gets a tight buffer. */
    int32_t actual_bytes = (*out_w) * (*out_h) * 3;
    if (actual_bytes > 0 && actual_bytes < alloc_w * alloc_h * 3) {
        uint8_t *tight = (uint8_t *)realloc(rgb, (size_t)actual_bytes);
        if (tight) rgb = tight;
    }

    *out_rgb = rgb;
    return 0;
}

NODUS_API void nodus_weight_image_free(uint8_t *rgb) {
    free(rgb);
}
