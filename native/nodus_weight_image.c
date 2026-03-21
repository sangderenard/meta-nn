/*
 * nodus_weight_image.c -- Weight-to-image renderer.
 *
 * Implements all three modes:
 *   parameter_groups    — square grid, 1 px per node group, nearest upscale
 *   architectural_tall   — layers as vertical columns, 1 px per unit
 *   architectural_wide   — transposed: layers as horizontal rows
 *   architectural_packed — packed layer boxes using a mean-width cap
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
static float minf(float a, float b) { return a < b ? a : b; }

static int maxi(int a, int b) { return a > b ? a : b; }
static int mini(int a, int b) { return a < b ? a : b; }

static int round_div_pos(int a, int b) {
    if (b <= 0) return a;
    return (a + (b / 2)) / b;
}

static uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static void hsv_to_rgb(float h_deg, float s, float v, uint8_t *r, uint8_t *g, uint8_t *b) {
    float h = fmodf(h_deg, 360.0f);
    if (h < 0.0f) h += 360.0f;
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    if (h < 60.0f)      { rr = c; gg = x; bb = 0.0f; }
    else if (h < 120.0f){ rr = x; gg = c; bb = 0.0f; }
    else if (h < 180.0f){ rr = 0.0f; gg = c; bb = x; }
    else if (h < 240.0f){ rr = 0.0f; gg = x; bb = c; }
    else if (h < 300.0f){ rr = x; gg = 0.0f; bb = c; }
    else                { rr = c; gg = 0.0f; bb = x; }
    *r = (uint8_t)clampf((rr + m) * 255.0f, 0.0f, 255.0f);
    *g = (uint8_t)clampf((gg + m) * 255.0f, 0.0f, 255.0f);
    *b = (uint8_t)clampf((bb + m) * 255.0f, 0.0f, 255.0f);
}

static void group_color_from_id(int group_id, uint8_t *r, uint8_t *g, uint8_t *b) {
    uint32_t hv = hash_u32((uint32_t)(group_id >= 0 ? group_id : -group_id));
    float t = (float)(hv & 0xFFFFU) / 65535.0f;
    /* Keep hues away from red; reserve red for megalayers. */
    float hue = 35.0f + (t * 280.0f);
    hsv_to_rgb(hue, 0.65f, 0.92f, r, g, b);
}

static void blend_pixel(uint8_t *rgb, int stride_w, int total_h,
                        int x, int y, uint8_t r, uint8_t g, uint8_t b, float alpha)
{
    if (x < 0 || x >= stride_w || y < 0 || y >= total_h) return;
    int off = (y * stride_w + x) * 3;
    float a = clampf(alpha, 0.0f, 1.0f);
    rgb[off + 0] = (uint8_t)clampf((1.0f - a) * (float)rgb[off + 0] + a * (float)r, 0.0f, 255.0f);
    rgb[off + 1] = (uint8_t)clampf((1.0f - a) * (float)rgb[off + 1] + a * (float)g, 0.0f, 255.0f);
    rgb[off + 2] = (uint8_t)clampf((1.0f - a) * (float)rgb[off + 2] + a * (float)b, 0.0f, 255.0f);
}

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

static void draw_rect_border(
        uint8_t *rgb, int stride_w, int total_h,
        int x0, int y0, int x1, int y1,
        uint8_t r, uint8_t g, uint8_t b)
{
    if (x1 < x0 || y1 < y0) return;
    for (int x = x0; x <= x1; x++) {
        set_pixel(rgb, stride_w, total_h, x, y0, r, g, b);
        set_pixel(rgb, stride_w, total_h, x, y1, r, g, b);
    }
    for (int y = y0; y <= y1; y++) {
        set_pixel(rgb, stride_w, total_h, x0, y, r, g, b);
        set_pixel(rgb, stride_w, total_h, x1, y, r, g, b);
    }
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

/* Nearest-neighbour resize to an arbitrary (non-integer) output size.
 * Maps each dst pixel back to the nearest src pixel using integer arithmetic
 * (no floating-point per-pixel work, identical output to a float NN filter).
 * dst must have room for dst_w * dst_h * 3 bytes. */
static void nn_resize_exact(const uint8_t *src, int src_w, int src_h,
                             uint8_t *dst, int dst_w, int dst_h)
{
    for (int y = 0; y < dst_h; y++) {
        int sy = (y * src_h) / dst_h;
        if (sy >= src_h) sy = src_h - 1;
        for (int x = 0; x < dst_w; x++) {
            int sx = (x * src_w) / dst_w;
            if (sx >= src_w) sx = src_w - 1;
            int si = (sy * src_w + sx) * 3;
            int di = (y  * dst_w + x)  * 3;
            dst[di]     = src[si];
            dst[di + 1] = src[si + 1];
            dst[di + 2] = src[si + 2];
        }
    }
}

static int measure_parameter_groups_dims(
        int32_t num_nodes,
        int32_t target_w,
        int32_t target_h,
        int32_t *out_w,
        int32_t *out_h)
{
    int side_px;
    if (!out_w || !out_h || num_nodes <= 0) return -1;
    side_px = maxi(8, maxi(target_w, target_h));
    *out_w = side_px;
    *out_h = side_px;
    return 0;
}

/* Shared packed-layout math used by BOTH measure and render. */
static int compute_packed_layout_from_units(
        const int32_t *unit_counts,
        int32_t num_layers,
        int32_t target_h,
        int32_t footer_h,
    int use_nonviolator_max,
        int *image_display_cols_out,
        int *data_flat_cols_out,
        int *is_mega_out,
        int *out_data_capacity_h90,
        int *out_data_raw_h,
        int *out_nonviolating_mean_feature_count)
{
    int data_target_h;
    int data_capacity_h90;
    int sum_nonviolating_features = 0;
    int max_nonviolating_features = 1;
    int count_nonviolating_layers = 0;
    int nonviolating_mean_feature_count;
    int data_raw_h;

    if (!unit_counts || num_layers <= 0 || !image_display_cols_out || !data_flat_cols_out || !is_mega_out
            || !out_data_capacity_h90 || !out_data_raw_h || !out_nonviolating_mean_feature_count) return -1;

    data_target_h = maxi(1, target_h - footer_h);
    data_capacity_h90 = maxi(1, (int)floorf(0.90f * (float)data_target_h));

    for (int li = 0; li < num_layers; li++) {
        int feature_count = maxi(1, unit_counts[li]);
        if (feature_count <= data_capacity_h90) {
            sum_nonviolating_features += feature_count;
            if (feature_count > max_nonviolating_features) max_nonviolating_features = feature_count;
            count_nonviolating_layers += 1;
        }
    }

    if (count_nonviolating_layers > 0) {
        if (use_nonviolator_max) {
            nonviolating_mean_feature_count = maxi(1, max_nonviolating_features);
        } else {
            nonviolating_mean_feature_count = maxi(1, sum_nonviolating_features / count_nonviolating_layers);
        }
    } else {
        nonviolating_mean_feature_count = maxi(1, data_capacity_h90);
    }

    data_raw_h = data_capacity_h90;
    for (int li = 0; li < num_layers; li++) {
        int feature_count = maxi(1, unit_counts[li]);
        int data_flat_cols = maxi(1, (int)ceilf((float)feature_count / (float)data_capacity_h90));
        int exceeds_90 = (feature_count > data_capacity_h90) ? 1 : 0;
        int image_display_cols = exceeds_90
            ? maxi(1, (int)ceilf((float)feature_count / (float)nonviolating_mean_feature_count))
            : data_flat_cols;
        int data_rows_used = maxi(1, (int)ceilf((float)feature_count / (float)image_display_cols));

        if (data_rows_used > data_raw_h) data_raw_h = data_rows_used;

        image_display_cols_out[li] = image_display_cols;
        data_flat_cols_out[li] = data_flat_cols;
        is_mega_out[li] = exceeds_90 ? 1 : 0;
    }

    *out_data_capacity_h90 = data_capacity_h90;
    *out_data_raw_h = data_raw_h;
    *out_nonviolating_mean_feature_count = nonviolating_mean_feature_count;
    return 0;
}

static int measure_architectural_dims(
        const int32_t *unit_counts,
        int32_t num_layers,
        int32_t target_w,
        int32_t target_h,
        int transpose,
        int pack_to_mean,
        int32_t *out_w,
        int32_t *out_h)
{
    int use_nonviolator_max = 1;
    int footer_h = 18;
    int max_unit_count = 1;
    int raw_h = 1, raw_w = 0;
    int *cols_per_layer = NULL;
    int ori_w, ori_h;
    int eff_w, eff_h;
    int li;
    if (!unit_counts || num_layers <= 0 || !out_w || !out_h) return -1;
    cols_per_layer = (int *)calloc((size_t)num_layers, sizeof(int));
    if (!cols_per_layer) return -1;
    for (li = 0; li < num_layers; li++) {
        if (unit_counts[li] > max_unit_count) max_unit_count = unit_counts[li];
    }

    if (pack_to_mean) {
        int packed_h90 = 1;
        int packed_raw_h = 1;
        int packed_nonviolating_mean_features = 1;
        int *is_mega = NULL;
        int *seam_cols = NULL;

        is_mega = (int *)calloc((size_t)num_layers, sizeof(int));
        seam_cols = (int *)calloc((size_t)num_layers + 1u, sizeof(int));
        if (!is_mega || !seam_cols) {
            free(is_mega);
            free(seam_cols);
            free(cols_per_layer);
            return -1;
        }

        if (compute_packed_layout_from_units(
                unit_counts,
                num_layers,
                target_h,
                footer_h,
            use_nonviolator_max,
                cols_per_layer,
                seam_cols, /* temporary scratch buffer, contents overwritten below */
                is_mega,
                &packed_h90,
                &packed_raw_h,
                &packed_nonviolating_mean_features) != 0) {
            free(is_mega);
            free(seam_cols);
            free(cols_per_layer);
            return -1;
        }

        raw_h = packed_raw_h;
        raw_w = 0;
        memset(seam_cols, 0, ((size_t)num_layers + 1u) * sizeof(int));
        for (li = 0; li < num_layers; li++) {
            if (is_mega[li]) {
                is_mega[li] = 1;
                if (seam_cols[li] < 1) seam_cols[li] = 1;
                if (seam_cols[li + 1] < 1) seam_cols[li + 1] = 1;
            }
        }

        raw_w += seam_cols[0];
        for (li = 0; li < num_layers; li++) {
            raw_w += cols_per_layer[li];
            raw_w += seam_cols[li + 1];
        }

        free(is_mega);
        free(seam_cols);
    } else {
        raw_h = (max_unit_count % 2 == 1) ? max_unit_count : (max_unit_count + 1);
        raw_w = 0;
        for (li = 0; li < num_layers; li++) {
            int uc = maxi(1, unit_counts[li]);
            int image_display_cols = maxi(1, (int)ceilf((float)uc / (float)raw_h));
            cols_per_layer[li] = image_display_cols;
            raw_w += image_display_cols;
        }
    }
    raw_w = maxi(1, raw_w);
    raw_h = maxi(1, raw_h);
    if ((raw_h % 2) == 0) raw_h += 1;
    if (transpose) {
        ori_w = raw_h;
        ori_h = raw_w;
    } else {
        ori_w = raw_w;
        ori_h = raw_h;
    }
    eff_w = maxi(target_w, ori_w);
    eff_h = maxi(target_h, ori_h + footer_h);
    *out_w = eff_w;
    *out_h = eff_h;
    free(cols_per_layer);
    return 0;
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
    int image_display_cols;  /* image-space columns occupied in tall mode */
    int data_box_cols;       /* data-space natural square-box columns */
    int data_flat_cols;      /* data-space columns needed at H90 floor */
    int is_mega;    /* packed layer was clamped to mean width */
    int group_id;   /* branch/group identifier for tinting */
    int x0;        /* raw grid column start */
    int x1;        /* raw grid column end */
    int y_top_raw;  /* topmost raw row used by this layer's pixels */
    int y_bot_raw;  /* bottommost raw row used by this layer's pixels */
    float max_abs_mean;
    float max_energy;
    float max_diff;
} LayerLayout;

static int render_architectural(
        const NodusWeightLayer *layers, int32_t num_layers,
        int32_t target_w, int32_t target_h,
        int transpose,  /* 0 = tall, 1 = wide */
        int pack_to_mean,
        const int32_t *group_ids,
        int scale_flags,  /* NODUS_WEIGHT_FLAG_* from high byte of mode */
        uint8_t *out_rgb, int32_t *out_w, int32_t *out_h)
{
    int use_nonviolator_max = 1;
    if (num_layers <= 0 || !layers) return -1;

    int footer_h = 18;

    /* Find max unit count across all layers. */
    int max_unit_count = 1;
    int flat_raw_h = 1;
    int *data_flat_cols_per_layer = NULL;
    int *image_display_cols_per_layer = NULL;
    int *packed_is_mega = NULL;
    for (int i = 0; i < num_layers; i++) {
        if (layers[i].unit_count > max_unit_count)
            max_unit_count = layers[i].unit_count;
    }
    /* Force odd for symmetric centering. */
    int raw_h = (max_unit_count % 2 == 1) ? max_unit_count : max_unit_count + 1;

    /* Per-layer layout. */
    LayerLayout *ll = (LayerLayout *)malloc((size_t)num_layers * sizeof(LayerLayout));
    if (!ll) return -1;

    /* Temp buffers for robust_scale per layer. */
    float *tmp_vals = (float *)malloc((size_t)(max_unit_count > 0 ? max_unit_count : 1) * sizeof(float));
    if (!tmp_vals) { free(ll); return -1; }

    if (pack_to_mean) {
        data_flat_cols_per_layer = (int *)calloc((size_t)num_layers, sizeof(int));
        image_display_cols_per_layer = (int *)calloc((size_t)num_layers, sizeof(int));
        packed_is_mega = (int *)calloc((size_t)num_layers, sizeof(int));
        if (!data_flat_cols_per_layer || !image_display_cols_per_layer || !packed_is_mega) {
            free(tmp_vals);
            free(data_flat_cols_per_layer);
            free(image_display_cols_per_layer);
            free(packed_is_mega);
            free(ll);
            return -1;
        }
    }

    if (pack_to_mean) {
        int packed_h90 = 1;
        int packed_raw_h = 1;
        int packed_nonviolating_mean_features = 1;
        int32_t *unit_counts = (int32_t *)calloc((size_t)num_layers, sizeof(int32_t));
        if (!unit_counts) {
            free(tmp_vals);
            free(data_flat_cols_per_layer);
            free(image_display_cols_per_layer);
            free(packed_is_mega);
            free(ll);
            return -1;
        }
        for (int li = 0; li < num_layers; li++) unit_counts[li] = layers[li].unit_count;

        if (compute_packed_layout_from_units(
                unit_counts,
                num_layers,
                target_h,
                footer_h,
            use_nonviolator_max,
                image_display_cols_per_layer,
                data_flat_cols_per_layer,
                packed_is_mega,
                &packed_h90,
                &packed_raw_h,
                &packed_nonviolating_mean_features) != 0) {
            free(unit_counts);
            free(tmp_vals);
            free(data_flat_cols_per_layer);
            free(image_display_cols_per_layer);
            free(packed_is_mega);
            free(ll);
            return -1;
        }
        free(unit_counts);
        flat_raw_h = packed_h90;
        raw_h = packed_raw_h;
    }

    int boundary_gap_cols = (pack_to_mean && !transpose) ? 2 : 0;
    int *seam_cols = NULL; /* size = num_layers + 1, reserved boundary bands */
    int raw_total_w = 0;
    for (int li = 0; li < num_layers; li++) {
        int uc = maxi(1, layers[li].unit_count);
        int image_display_cols = 0;
        if (pack_to_mean) {
            int data_box_cols = maxi(1, (int)ceilf(sqrtf((float)uc)));
            int data_flat_cols = data_flat_cols_per_layer[li];
            image_display_cols = image_display_cols_per_layer[li];
            ll[li].data_box_cols = data_box_cols;
            ll[li].data_flat_cols = data_flat_cols;
            ll[li].is_mega = packed_is_mega[li];
        } else {
            image_display_cols = maxi(1, (int)ceilf((float)uc / (float)raw_h));
            ll[li].data_box_cols = image_display_cols;
            ll[li].data_flat_cols = image_display_cols;
            ll[li].is_mega = 0;
        }
        ll[li].unit_count = layers[li].unit_count;
        ll[li].image_display_cols = image_display_cols;
        ll[li].group_id = group_ids ? (int)group_ids[li] : 0;
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

    /* Compute all required boundary bands before any x allocation:
       - mega layers need left/right seam space for red border
       - branch/group spans need seam space at span boundaries */
    if (boundary_gap_cols > 0) {
        seam_cols = (int *)calloc((size_t)num_layers + 1u, sizeof(int));
        if (!seam_cols) {
            free(data_flat_cols_per_layer);
            free(image_display_cols_per_layer);
            free(packed_is_mega);
            free(ll);
            return -1;
        }

        for (int li = 0; li < num_layers; li++) {
            if (ll[li].is_mega) {
                if (seam_cols[li] < 1) seam_cols[li] = 1;
                if (seam_cols[li + 1] < 1) seam_cols[li + 1] = 1;
            }
        }

        {
            int group_count_layout = 0;
            int li = 0;
            while (li < num_layers) {
                int gid = ll[li].group_id;
                int end = li;
                group_count_layout++;
                while ((end + 1) < num_layers && ll[end + 1].group_id == gid) end++;
                li = end + 1;
            }
            if (group_count_layout > 1) {
                li = 0;
                while (li < num_layers) {
                    int gid = ll[li].group_id;
                    int start = li;
                    int end = li;
                    while ((end + 1) < num_layers && ll[end + 1].group_id == gid) end++;
                    if (seam_cols[start] < boundary_gap_cols) seam_cols[start] = boundary_gap_cols;
                    if (seam_cols[end + 1] < boundary_gap_cols) seam_cols[end + 1] = boundary_gap_cols;
                    li = end + 1;
                }
            }
        }
    }

    /* Assign layer x ranges from precomputed seam bands. */
    raw_total_w = 0;
    if (seam_cols) raw_total_w += seam_cols[0];
    for (int li = 0; li < num_layers; li++) {
        ll[li].x0 = raw_total_w;
        ll[li].x1 = raw_total_w + ll[li].image_display_cols;
        raw_total_w += ll[li].image_display_cols;
        if (seam_cols) raw_total_w += seam_cols[li + 1];
    }

    raw_h = maxi(1, raw_h);
    if ((raw_h % 2) == 0) raw_h += 1;

    int raw_w = maxi(1, raw_total_w);

    /* Build raw grid (1 px per unit). */
    uint8_t *raw = (uint8_t *)malloc((size_t)raw_h * (size_t)raw_w * 3);
    if (!raw) { free(seam_cols); free(data_flat_cols_per_layer); free(image_display_cols_per_layer); free(packed_is_mega); free(ll); return -1; }
    fill_bg(raw, raw_w, raw_h);

    for (int li = 0; li < num_layers; li++) {
        int n = layers[li].unit_count;
        const NodusWeightUnit *units = layers[li].units;
        int row_capacity = raw_h;
        if (pack_to_mean) {
            row_capacity = maxi(1, (int)ceilf((float)maxi(1, n) / (float)maxi(1, ll[li].image_display_cols)));
            row_capacity = mini(raw_h, row_capacity);
        }
        /* Box interior is row_capacity; border is drawn 1px outside by draw_rect_border. */
        ll[li].y_top_raw = (raw_h - row_capacity) / 2;
        ll[li].y_bot_raw = ll[li].y_top_raw + row_capacity - 1;
        for (int idx = 0; idx < n; idx++) {
            int col = idx / row_capacity;
            int pos_in_col = idx % row_capacity;
            int units_in_col = mini(row_capacity, maxi(1, n - col * row_capacity));
            int y_offset = (raw_h - units_in_col) / 2;
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
        if (!oriented) { free(raw); free(seam_cols); free(data_flat_cols_per_layer); free(image_display_cols_per_layer); free(packed_is_mega); free(ll); return -1; }
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
    int avail_w = maxi(1, eff_w);
    int avail_h = maxi(1, data_h);

    /* Compute scale according to the requested mode flag.
     *
     * fscale is kept as float for overlay coordinate math; scaled_w/h are
     * the actual pixel dimensions of the upscaled image.
     *
     *  default (0)              : integer gate — largest whole factor ≥ 1
     *  NODUS_WEIGHT_FLAG_NO_UPSCALE  : never scale; centre at 1×1
     *  NODUS_WEIGHT_FLAG_FRACTIONAL_NN : float scale that fills available
     *                                    space exactly, NN interpolation
     */
    float fscale;
    int   scaled_w, scaled_h;

    if (scale_flags & NODUS_WEIGHT_FLAG_NO_UPSCALE) {
        fscale   = 1.0f;
        scaled_w = ori_w;
        scaled_h = ori_h;
    } else if (scale_flags & NODUS_WEIGHT_FLAG_FRACTIONAL_NN) {
        fscale = minf((float)avail_h / (float)ori_h,
                      (float)avail_w / (float)ori_w);
        if (fscale < 1.0f) fscale = 1.0f;
        scaled_w = (int)(ori_w * fscale);
        scaled_h = (int)(ori_h * fscale);
        if (scaled_w < 1) scaled_w = 1;
        if (scaled_h < 1) scaled_h = 1;
    } else {
        /* Integer gate mode (default). */
        int scale_i = maxi(1, mini(avail_h / ori_h, avail_w / ori_w));
        fscale   = (float)scale_i;
        scaled_w = ori_w * scale_i;
        scaled_h = ori_h * scale_i;
    }

    /* Upscale / resize. */
    uint8_t *scaled;
    int is_exact_resize = (scale_flags & NODUS_WEIGHT_FLAG_FRACTIONAL_NN) != 0
                          && (scaled_w != ori_w || scaled_h != ori_h);
    int is_int_upscale  = !(scale_flags & (NODUS_WEIGHT_FLAG_FRACTIONAL_NN |
                                           NODUS_WEIGHT_FLAG_NO_UPSCALE))
                          && (scaled_w != ori_w || scaled_h != ori_h);

    if (is_exact_resize) {
        scaled = (uint8_t *)malloc((size_t)scaled_w * (size_t)scaled_h * 3);
        if (!scaled) { free(oriented); free(seam_cols); free(data_flat_cols_per_layer); free(image_display_cols_per_layer); free(packed_is_mega); free(ll); return -1; }
        nn_resize_exact(oriented, ori_w, ori_h, scaled, scaled_w, scaled_h);
        free(oriented);
    } else if (is_int_upscale) {
        int scale_i = (int)fscale;
        scaled = (uint8_t *)malloc((size_t)scaled_w * (size_t)scaled_h * 3);
        if (!scaled) { free(oriented); free(seam_cols); free(data_flat_cols_per_layer); free(image_display_cols_per_layer); free(packed_is_mega); free(ll); return -1; }
        nn_upscale(oriented, ori_w, ori_h, scaled, scale_i, scale_i);
        free(oriented);
    } else {
        scaled = oriented;
    }

    /* Black canvas of exactly (eff_h x eff_w). */
    fill_black(out_rgb, eff_w, eff_h);

    /* Centre the scaled grid in the data area. */
    int y0 = (avail_h - scaled_h) / 2;
    int x0 = (avail_w - scaled_w) / 2;
    int grid_x0 = x0;
    int grid_x1 = x0 + scaled_w - 1;
    int grid_y0 = y0;
    int grid_y1 = y0 + scaled_h - 1;
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

    /* Packed mode overlays: group tints, megalayer red tint, and nested borders.
       Draw order is intentional: mega borders are inset, group borders are last. */
    if (pack_to_mean && !transpose) {
        int *group_start = (int *)calloc((size_t)num_layers, sizeof(int));
        int *group_end = (int *)calloc((size_t)num_layers, sizeof(int));
        int *group_gid = (int *)calloc((size_t)num_layers, sizeof(int));
        int group_count = 0;
        if (!group_start || !group_end || !group_gid) {
            free(group_start); free(group_end); free(group_gid);
            free(seam_cols);
            free(data_flat_cols_per_layer);
            free(image_display_cols_per_layer);
            free(packed_is_mega);
            free(ll);
            return -1;
        }

        /* Build contiguous spans by group id to represent branch partitions. */
        {
            int li = 0;
            while (li < num_layers) {
                int gid = ll[li].group_id;
                int start = li;
                int end = li;
                while ((end + 1) < num_layers && ll[end + 1].group_id == gid) end++;
                group_start[group_count] = start;
                group_end[group_count] = end;
                group_gid[group_count] = gid;
                group_count++;
                li = end + 1;
            }
        }

        {
            int enable_group_overlay = (group_count > 1) ? 1 : 0;

            /* Pass 1: tint whole branch/group spans (only when actual branching exists). */
            if (enable_group_overlay) {
                for (int gi = 0; gi < group_count; gi++) {
                    int li0 = group_start[gi];
                    int li1 = group_end[gi];
                    int gx0 = x0 + (int)(ll[li0].x0 * fscale);
                    int gx1 = x0 + (int)(ll[li1].x1 * fscale);
                    int y_start = eff_h, y_end = -1;
                    for (int k = li0; k <= li1; k++) {
                        int ly_top = y0 + (int)(ll[k].y_top_raw * fscale);
                        int ly_bot = y0 + (int)((ll[k].y_bot_raw + 1) * fscale) - 1;
                        if (ly_top < y_start) y_start = ly_top;
                        if (ly_bot > y_end)   y_end   = ly_bot;
                    }
                    uint8_t gr, gg, gb;
                    if (gx1 <= gx0) gx1 = gx0 + 1;
                    gx0 = maxi(0, gx0);
                    gx1 = mini(eff_w, gx1);
                    if (gx0 >= gx1) continue;

                    group_color_from_id(group_gid[gi], &gr, &gg, &gb);
                    for (int y = y_start; y <= y_end; y++) {
                        for (int x = gx0; x < gx1; x++) {
                            blend_pixel(out_rgb, eff_w, eff_h, x, y, gr, gg, gb, 0.12f);
                        }
                    }
                }
            }

            /* Pass 2: mega tint + dark-grey border ring using inter-layer boundary bands. */
            for (int li = 0; li < num_layers; li++) {
                int lx0 = x0 + (int)(ll[li].x0 * fscale);
                int lx1 = x0 + (int)(ll[li].x1 * fscale);
                int y_start = y0 + (int)(ll[li].y_top_raw * fscale);
                int y_end   = y0 + (int)((ll[li].y_bot_raw + 1) * fscale) - 1;
                int left_seam = seam_cols ? seam_cols[li] : 0;
                int right_seam = seam_cols ? seam_cols[li + 1] : 0;
                if (!ll[li].is_mega) continue;
                if (lx1 <= lx0) lx1 = lx0 + 1;
                lx0 = maxi(0, lx0);
                lx1 = mini(eff_w, lx1);
                if (lx0 >= lx1) continue;

                for (int y = y_start; y <= y_end; y++) {
                    for (int x = lx0; x < lx1; x++) {
                        blend_pixel(out_rgb, eff_w, eff_h, x, y, 64, 64, 68, 0.16f);
                    }
                }

                {
                    int x_left = maxi(0, lx0 - (left_seam > 0 ? 1 : 0));
                    int x_right = mini(eff_w - 1, (right_seam > 0) ? lx1 : (lx1 - 1));
                    int y_top = maxi(0, y_start - 1);
                    int y_bottom = mini(data_h - 1, y_end + 1);
                    /* Avoid rendering 1px seam-lines as fake borders for tiny boxes. */
                    if ((x_right - x_left) >= 2 && (y_bottom - y_top) >= 2) {
                        draw_rect_border(out_rgb, eff_w, eff_h, x_left, y_top, x_right, y_bottom, 72, 72, 76);
                    }
                }
            }

            /* Pass 3: group borders last; only when branch partitions exist. */
            if (enable_group_overlay) {
                for (int gi = 0; gi < group_count; gi++) {
                    int li0 = group_start[gi];
                    int li1 = group_end[gi];
                    int gx0 = x0 + (int)(ll[li0].x0 * fscale);
                    int gx1 = x0 + (int)(ll[li1].x1 * fscale);
                    int y_start = eff_h, y_end = -1;
                    for (int k = li0; k <= li1; k++) {
                        int ly_top = y0 + (int)(ll[k].y_top_raw * fscale);
                        int ly_bot = y0 + (int)((ll[k].y_bot_raw + 1) * fscale) - 1;
                        if (ly_top < y_start) y_start = ly_top;
                        if (ly_bot > y_end)   y_end   = ly_bot;
                    }
                    int left_seam = seam_cols ? seam_cols[li0] : 0;
                    int right_seam = seam_cols ? seam_cols[li1 + 1] : 0;
                    uint8_t gr, gg, gb;
                    if (gx1 <= gx0) gx1 = gx0 + 1;
                    gx0 = maxi(0, gx0);
                    gx1 = mini(eff_w, gx1);
                    if (gx0 >= gx1) continue;

                    group_color_from_id(group_gid[gi], &gr, &gg, &gb);
                    draw_rect_border(
                        out_rgb,
                        eff_w,
                        eff_h,
                        maxi(0, gx0 - (left_seam > 0 ? 1 : 0)),
                        maxi(0, y_start - 2),
                        mini(eff_w - 1, (right_seam > 0) ? gx1 : (gx1 - 1)),
                        mini(data_h - 1, y_end + 1),
                        (uint8_t)clampf((float)gr * 0.95f, 0.0f, 255.0f),
                        (uint8_t)clampf((float)gg * 0.95f, 0.0f, 255.0f),
                        (uint8_t)clampf((float)gb * 0.95f, 0.0f, 255.0f)
                    );
                }
            }
        }

        free(group_start);
        free(group_end);
        free(group_gid);
    }

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
    /* Tick marks at each layer center (use fscale for sub-integer alignment). */
    for (int li = 0; li < num_layers; li++) {
        int lx0 = x0 + (int)(ll[li].x0 * fscale);
        int lx1 = x0 + (int)(ll[li].x1 * fscale);
        if (lx1 <= lx0) lx1 = lx0 + 1;
        int cx = (lx0 + lx1) / 2;
        for (int y = data_h; y < eff_h; y++) {
            set_pixel(out_rgb, eff_w, eff_h, cx, y, 76, 84, 96);
        }
    }

    free(ll);
    free(seam_cols);
    free(data_flat_cols_per_layer);
    free(image_display_cols_per_layer);
    free(packed_is_mega);

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

    int pure_mode   = mode & 0xFF;
    int scale_flags = mode >> 8;

    switch (pure_mode) {
    case NODUS_WEIGHT_MODE_PARAMETER_GROUPS:
        if (!nodes || num_nodes <= 0) return -1;
        return render_parameter_groups(nodes, num_nodes,
                                       target_w, target_h,
                                       out_rgb, out_w, out_h);

    case NODUS_WEIGHT_MODE_ARCHITECTURAL_TALL:
        if (!layers || num_layers <= 0) return -1;
        return render_architectural(layers, num_layers,
                                    target_w, target_h,
                                    0, 0, NULL, scale_flags, out_rgb, out_w, out_h);

    case NODUS_WEIGHT_MODE_ARCHITECTURAL_WIDE:
        if (!layers || num_layers <= 0) return -1;
        return render_architectural(layers, num_layers,
                                    target_w, target_h,
                                    1, 0, NULL, scale_flags, out_rgb, out_w, out_h);

    case NODUS_WEIGHT_MODE_ARCHITECTURAL_PACKED:
        if (!layers || num_layers <= 0) return -1;
        return render_architectural(layers, num_layers,
                                    target_w, target_h,
                                    0, 1, NULL, scale_flags, out_rgb, out_w, out_h);

    default:
        return -1;
    }
}

NODUS_API int nodus_weight_image_measure(
        int                         mode,
        const NodusWeightLayer     *layers,
        int32_t                     num_layers,
        const NodusWeightNodeGroup *nodes,
        int32_t                     num_nodes,
        int32_t                     target_w,
        int32_t                     target_h,
        int32_t                    *out_w,
        int32_t                    *out_h)
{
    target_w = maxi(8, target_w);
    target_h = maxi(8, target_h);
    int pure_mode = mode & 0xFF; /* strip scale flags — measure output size is invariant */
    switch (pure_mode) {
    case NODUS_WEIGHT_MODE_PARAMETER_GROUPS:
        if (!nodes || num_nodes <= 0) return -1;
        return measure_parameter_groups_dims(num_nodes, target_w, target_h, out_w, out_h);

    case NODUS_WEIGHT_MODE_ARCHITECTURAL_TALL: {
        int32_t *unit_counts = NULL;
        int rc;
        if (!layers || num_layers <= 0) return -1;
        unit_counts = (int32_t *)calloc((size_t)num_layers, sizeof(int32_t));
        if (!unit_counts) return -1;
        for (int32_t i = 0; i < num_layers; i++) unit_counts[i] = layers[i].unit_count;
        rc = measure_architectural_dims(unit_counts, num_layers, target_w, target_h, 0, 0, out_w, out_h);
        free(unit_counts);
        return rc;
    }

    case NODUS_WEIGHT_MODE_ARCHITECTURAL_WIDE: {
        int32_t *unit_counts = NULL;
        int rc;
        if (!layers || num_layers <= 0) return -1;
        unit_counts = (int32_t *)calloc((size_t)num_layers, sizeof(int32_t));
        if (!unit_counts) return -1;
        for (int32_t i = 0; i < num_layers; i++) unit_counts[i] = layers[i].unit_count;
        rc = measure_architectural_dims(unit_counts, num_layers, target_w, target_h, 1, 0, out_w, out_h);
        free(unit_counts);
        return rc;
    }

    case NODUS_WEIGHT_MODE_ARCHITECTURAL_PACKED: {
        int32_t *unit_counts = NULL;
        int rc;
        if (!layers || num_layers <= 0) return -1;
        unit_counts = (int32_t *)calloc((size_t)num_layers, sizeof(int32_t));
        if (!unit_counts) return -1;
        for (int32_t i = 0; i < num_layers; i++) unit_counts[i] = layers[i].unit_count;
        rc = measure_architectural_dims(unit_counts, num_layers, target_w, target_h, 0, 1, out_w, out_h);
        free(unit_counts);
        return rc;
    }

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

NODUS_API int nodus_weight_image_measure_from_state_dict(
        const char *const  *param_names,
        const float *const *param_data,
        const int32_t      *param_numel,
        const int32_t      *param_shape0,
        int32_t             num_params,
        int                 mode,
        int32_t             target_w,
        int32_t             target_h,
        int32_t            *out_w,
        int32_t            *out_h)
{
    NodeAccum *nodes_acc = NULL;
    int num_groups = 0;
    int32_t *shape0_buf = NULL;
    int32_t *unit_counts = NULL;
    int rc = -1;

    if (!param_names || !param_data || !param_numel || !param_shape0 || num_params <= 0 || !out_w || !out_h) {
        return -1;
    }
    (void)param_data;
    (void)param_numel;

    target_w = maxi(8, target_w);
    target_h = maxi(8, target_h);

    nodes_acc = (NodeAccum *)calloc((size_t)num_params, sizeof(NodeAccum));
    if (!nodes_acc) return -1;

    for (int pi = 0; pi < num_params; pi++) {
        char key[256];
        int gi = -1;
        param_node_key(param_names[pi], key, 256);
        for (int g = 0; g < num_groups; g++) {
            if (strcmp(nodes_acc[g].name, key) == 0) {
                gi = g;
                break;
            }
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

    {
        int pure_mode_m = mode & 0xFF;  /* scale flags don't affect measured dimensions */
        if (pure_mode_m == NODUS_WEIGHT_MODE_PARAMETER_GROUPS) {
            rc = measure_parameter_groups_dims(num_groups, target_w, target_h, out_w, out_h);
            free(nodes_acc);
            return rc;
        }

        shape0_buf = (int32_t *)malloc((size_t)num_params * sizeof(int32_t));
        unit_counts = (int32_t *)calloc((size_t)num_groups, sizeof(int32_t));
        if (!shape0_buf || !unit_counts) goto cleanup;

        for (int g = 0; g < num_groups; g++) {
            int np = nodes_acc[g].num_params;
            for (int j = 0; j < np; j++) {
                shape0_buf[j] = param_shape0[nodes_acc[g].param_indices[j]];
            }
            unit_counts[g] = choose_unit_count(shape0_buf, np);
        }

        rc = measure_architectural_dims(
            unit_counts,
            num_groups,
            target_w,
            target_h,
            (pure_mode_m == NODUS_WEIGHT_MODE_ARCHITECTURAL_WIDE) ? 1 : 0,
            (pure_mode_m == NODUS_WEIGHT_MODE_ARCHITECTURAL_PACKED) ? 1 : 0,
            out_w,
            out_h
        );
    }

cleanup:
    free(unit_counts);
    free(shape0_buf);
    free(nodes_acc);
    return rc;
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

    int pure_mode   = mode & 0xFF;
    int scale_flags = mode >> 8;

    /* ---- PARAMETER_GROUPS mode: aggregate per node group ---- */
    if (pure_mode == NODUS_WEIGHT_MODE_PARAMETER_GROUPS) {
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
    int32_t *group_ids = (int32_t *)calloc((size_t)num_groups, sizeof(int32_t));
    if (!layers) { free(nodes_acc); return -1; }
    if (!group_ids) { free(layers); free(nodes_acc); return -1; }

    for (int g = 0; g < num_groups; g++) {
        const char *gname = nodes_acc[g].name;
        int gid = 0;
        for (int i = 0; gname[i] != '\0' && gname[i] != '.'; i++) {
            gid = ((gid * 131) + (unsigned char)gname[i]) & 0x7fffffff;
        }
        group_ids[g] = (int32_t)gid;

        int uc = nodes_acc[g].unit_count;
        layers[g].unit_count = uc;
        layers[g].units = (NodusWeightUnit *)calloc((size_t)uc, sizeof(NodusWeightUnit));
        if (!layers[g].units) {
            for (int k = 0; k < g; k++) free(layers[k].units);
            free(group_ids); free(layers); free(nodes_acc);
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
            free(group_ids); free(layers); free(nodes_acc);
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
    int32_t *unit_counts_tmp = (int32_t *)calloc((size_t)num_groups, sizeof(int32_t));
    int32_t alloc_w = 0;
    int32_t alloc_h = 0;
    if (!unit_counts_tmp) {
        for (int g = 0; g < num_groups; g++) free(layers[g].units);
        free(group_ids);
        free(layers);
        return -1;
    }
    for (int g = 0; g < num_groups; g++) unit_counts_tmp[g] = (int32_t)layers[g].unit_count;
    if (measure_architectural_dims(
            unit_counts_tmp,
            num_groups,
            target_w,
            target_h,
            (pure_mode == NODUS_WEIGHT_MODE_ARCHITECTURAL_WIDE) ? 1 : 0,
            (pure_mode == NODUS_WEIGHT_MODE_ARCHITECTURAL_PACKED) ? 1 : 0,
            &alloc_w,
            &alloc_h) != 0) {
        free(unit_counts_tmp);
        for (int g = 0; g < num_groups; g++) free(layers[g].units);
        free(group_ids);
        free(layers);
        return -1;
    }
    free(unit_counts_tmp);

    uint8_t *rgb = (uint8_t *)malloc((size_t)alloc_w * (size_t)alloc_h * 3);
    if (!rgb) {
        for (int g = 0; g < num_groups; g++) free(layers[g].units);
        free(group_ids);
        free(layers);
        return -1;
    }

    int transpose    = (pure_mode == NODUS_WEIGHT_MODE_ARCHITECTURAL_WIDE) ? 1 : 0;
    int pack_to_mean = (pure_mode == NODUS_WEIGHT_MODE_ARCHITECTURAL_PACKED) ? 1 : 0;
    int rc = render_architectural(layers, num_groups,
                                  target_w, target_h,
                                  transpose, pack_to_mean, group_ids,
                                  scale_flags, rgb, out_w, out_h);

    for (int g = 0; g < num_groups; g++) free(layers[g].units);
    free(group_ids);
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
