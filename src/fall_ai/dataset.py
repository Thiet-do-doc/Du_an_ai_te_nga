"""Khuyến: quản lý manifest và chia tập trước khi tạo cửa sổ."""


def build_manifest(raw_dir, output_csv):
    raise NotImplementedError("Kiểm kê video; xuất schema tại docs/CONTRACTS.md")


def split_manifest(manifest_csv, output_csv):
    raise NotImplementedError("Chia theo subject_id/sự kiện; không chia ngẫu nhiên frame")
