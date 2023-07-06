colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def hex_to_rgba(h):
    tuple_rgb = tuple(float(int(h[i:i+2], 16)) / 255.0 for i in (0, 2, 4))
    list_rgba = list(tuple_rgb)
    list_rgba.append(1.0)
    tuple_rgba = tuple(list_rgba)
    return tuple_rgba


color_0 = hex_to_rgba(colors[0][1:])
color_1 = hex_to_rgba(colors[1][1:])
