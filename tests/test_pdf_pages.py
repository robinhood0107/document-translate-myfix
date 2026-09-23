from __future__ import annotations

import errno
import os
import shutil
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import img2pdf
import imkit as imk
import numpy as np
import pikepdf
from PIL import Image, features

from modules.utils import pdf_pages as pdf_pages_module
from modules.utils.file_handler import FileHandler
from modules.utils.exceptions import OperationCancelledError
from modules.ocr.base import OCREngine
from modules.utils.archives import list_archive_image_entries
from modules.utils.pdf_pages import (
    PDF_RENDER_LONG_SIDE_LIMIT,
    PDF_RENDER_PIXEL_LIMIT,
    _PDF_CACHE,
    _page_objects,
    _requested_dpi,
    PdfImportError,
    _render_dimensions,
    close_pdf_cache,
    materialize_page,
    materialize_transaction,
    scan_pdf,
    validate_materialized_page,
)


def _icc_profile_stream(pdf: pikepdf.Pdf):
    profile = pdf.make_stream(b"\0" * 128)
    profile["/N"] = 3
    return profile


def _prepend_content(pdf: pikepdf.Pdf, prefix: bytes, suffix: bytes) -> None:
    page = pdf.pages[0]
    page.obj["/Contents"] = pikepdf.Array(
        [pdf.make_stream(prefix), page.obj["/Contents"], pdf.make_stream(suffix)]
    )


def _append_duplicate_image(pdf: pikepdf.Pdf) -> None:
    page = pdf.pages[0]
    page.obj["/Contents"] = pikepdf.Array(
        [page.obj["/Contents"], page.obj["/Contents"]]
    )


class PdfPagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="ct_pdf_test_")

    def tearDown(self) -> None:
        close_pdf_cache()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.temp_dir, name)

    def _image_pdf(
        self,
        *,
        image_format: str = "JPEG",
        count: int = 1,
        mode: str = "RGB",
    ) -> tuple[str, list[str]]:
        image_paths: list[str] = []
        suffix = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "JPEG2000": ".jp2",
            "TIFF": ".tiff",
        }[image_format]
        for index in range(count):
            image_path = self._path(f"source_{index}{suffix}")
            color = (20 + index, 40 + index, 60 + index) if mode == "RGB" else 80 + index
            image = Image.new(mode, (120, 80), color)
            save_args = {"quality": 92} if image_format == "JPEG" else {}
            if image_format == "TIFF":
                save_args["compression"] = "tiff_adobe_deflate"
            image.save(image_path, image_format, **save_args)
            image_paths.append(image_path)
        pdf_path = self._path("source.pdf")
        Path(pdf_path).write_bytes(img2pdf.convert(image_paths))
        return pdf_path, image_paths

    def _rewrite_pdf(self, source: str, destination: str, mutator) -> str:
        with pikepdf.open(source) as pdf:
            mutator(pdf)
            pdf.save(destination)
        return destination

    def _assert_working_image_contract(self, path: str) -> None:
        image = imk.read_image(path)
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)

        class RecordingOcr(OCREngine):
            def initialize(self, **kwargs) -> None:
                pass

            def process_image(self, img, blk_list):
                self.received = img
                return blk_list

        adapter = RecordingOcr()
        adapter.process_image(image, [])
        self.assertEqual(adapter.received.shape, image.shape)
        self.assertEqual(adapter.received.dtype, np.uint8)

    def test_jpeg_native_copy_preserves_encoded_and_rgb_hashes(self) -> None:
        pdf_path, images = self._image_pdf(image_format="JPEG")
        _identity, plans = scan_pdf(pdf_path)

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].strategy, "native")
        self.assertEqual(plans[0].extension, ".jpg")

        output = self._path("page.jpg")
        materialize_page(pdf_path, plans[0], output)
        self.assertEqual(Path(output).read_bytes(), Path(images[0]).read_bytes())
        self.assertTrue(validate_materialized_page(plans[0], output))
        self._assert_working_image_contract(output)

    def test_native_oracle_failure_forces_render_without_lossless_downgrade(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")

        with mock.patch(
            "modules.utils.pdf_pages._native_candidate", return_value=None
        ), mock.patch(
            "modules.utils.pdf_pages._lossless_candidate",
            side_effect=AssertionError("native formats must not downgrade to lossless"),
        ):
            plan = scan_pdf(pdf_path)[1][0]

        self.assertEqual(plan.strategy, "render")

    def test_jpeg2000_native_copy_preserves_boxed_codestream(self) -> None:
        if not features.check("jpg_2000"):
            self.skipTest("Pillow JPEG2000 support is unavailable")
        pdf_path, images = self._image_pdf(image_format="JPEG2000")
        _identity, plans = scan_pdf(pdf_path)

        self.assertEqual(plans[0].strategy, "native")
        self.assertEqual(plans[0].extension, ".jp2")
        output = self._path("page.jp2")
        materialize_page(pdf_path, plans[0], output)
        self.assertEqual(Path(output).read_bytes(), Path(images[0]).read_bytes())
        self._assert_working_image_contract(output)

    def test_jpeg2000_without_pdf_colorspace_can_still_be_native(self) -> None:
        if not features.check("jpg_2000"):
            self.skipTest("Pillow JPEG2000 support is unavailable")
        source, _images = self._image_pdf(image_format="JPEG2000")
        altered = self._path("jp2_without_colorspace.pdf")

        def remove_colorspace(pdf) -> None:
            stream = next(iter(pdf.pages[0].images.values()))
            del stream["/ColorSpace"]

        self._rewrite_pdf(source, altered, remove_colorspace)
        _identity, plans = scan_pdf(altered)
        self.assertEqual(plans[0].strategy, "native")
        self.assertEqual(plans[0].extension, ".jp2")

    def test_raw_j2k_native_copy_preserves_codestream(self) -> None:
        if not features.check("jpg_2000"):
            self.skipTest("Pillow JPEG2000 support is unavailable")
        image_path = self._path("source.j2k")
        Image.new("RGB", (120, 80), (20, 40, 60)).save(
            image_path, "JPEG2000", no_jp2=True
        )
        pdf_path = self._path("raw_j2k.pdf")
        Path(pdf_path).write_bytes(img2pdf.convert(image_path))

        _identity, plans = scan_pdf(pdf_path)
        self.assertEqual(plans[0].strategy, "native")
        self.assertEqual(plans[0].extension, ".j2k")
        output = self._path("page.j2k")
        materialize_page(pdf_path, plans[0], output)
        self.assertEqual(Path(output).read_bytes(), Path(image_path).read_bytes())
        self._assert_working_image_contract(output)

    def test_flate_rgb_is_lossless_png(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="PNG")
        _identity, plans = scan_pdf(pdf_path)

        self.assertEqual(plans[0].strategy, "lossless")
        self.assertEqual(plans[0].extension, ".png")
        output = self._path("page.png")
        materialize_page(pdf_path, plans[0], output)
        with Image.open(output) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (120, 80))
        self.assertTrue(validate_materialized_page(plans[0], output))
        self._assert_working_image_contract(output)

    def test_flate_cmyk_is_lossless_tiff(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="TIFF", mode="CMYK")
        _identity, plans = scan_pdf(pdf_path)

        self.assertEqual(plans[0].strategy, "lossless")
        self.assertEqual(plans[0].extension, ".tiff")
        output = self._path("page.tiff")
        materialize_page(pdf_path, plans[0], output)
        with Image.open(output) as image:
            self.assertEqual(image.mode, "CMYK")
            self.assertEqual(image.size, (120, 80))
        self.assertTrue(validate_materialized_page(plans[0], output))
        self._assert_working_image_contract(output)

    def test_alpha_and_16_bit_images_force_render(self) -> None:
        alpha_path = self._path("alpha.png")
        Image.new("RGBA", (64, 48), (1, 2, 3, 128)).save(alpha_path)
        alpha_pdf = self._path("alpha_image.pdf")
        Path(alpha_pdf).write_bytes(img2pdf.convert(alpha_path))
        self.assertEqual(scan_pdf(alpha_pdf)[1][0].strategy, "render")
        close_pdf_cache(alpha_pdf)

        high_bit_path = self._path("high_bit.png")
        samples = b"".join(value.to_bytes(2, "little") for value in range(64 * 48))
        Image.frombytes("I;16", (64, 48), samples).save(high_bit_path)
        high_bit_pdf = self._path("high_bit.pdf")
        Path(high_bit_pdf).write_bytes(img2pdf.convert(high_bit_path))
        high_bit_plan = scan_pdf(high_bit_pdf)[1][0]
        self.assertEqual(high_bit_plan.strategy, "render")
        high_bit_output = self._path("high_bit_render.png")
        materialize_page(high_bit_pdf, high_bit_plan, high_bit_output)
        self._assert_working_image_contract(high_bit_output)

    def test_rotation_forces_pdfium_rgb_render(self) -> None:
        source, _images = self._image_pdf(image_format="JPEG")
        rotated = self._path("rotated.pdf")

        def rotate(pdf) -> None:
            pdf.pages[0].obj["/Rotate"] = 90

        self._rewrite_pdf(source, rotated, rotate)
        _identity, plans = scan_pdf(rotated)
        self.assertEqual(plans[0].strategy, "render")
        self.assertEqual(plans[0].extension, ".png")

        output = self._path("rotated.png")
        materialize_page(rotated, plans[0], output)
        with Image.open(output) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, plans[0].render_size)

    def test_nonopaque_extgstate_forces_render(self) -> None:
        source, _images = self._image_pdf(image_format="JPEG")
        altered = self._path("alpha.pdf")

        def add_alpha(pdf) -> None:
            page = pdf.pages[0]
            resources = page.obj["/Resources"]
            resources["/ExtGState"] = pikepdf.Dictionary(
                GSalpha=pikepdf.Dictionary(
                    Type=pikepdf.Name("/ExtGState"),
                    BM=pikepdf.Name("/Normal"),
                    CA=1,
                    ca=0.99,
                )
            )
            original = page.obj["/Contents"]
            prefix = pdf.make_stream(b"q /GSalpha gs\n")
            suffix = pdf.make_stream(b"Q\n")
            page.obj["/Contents"] = pikepdf.Array([prefix, original, suffix])

        self._rewrite_pdf(source, altered, add_alpha)
        _identity, plans = scan_pdf(altered)
        self.assertEqual(plans[0].strategy, "render")

    def test_only_opaque_normal_extgstate_is_direct(self) -> None:
        source, _images = self._image_pdf(image_format="JPEG")

        def write_variant(name: str, state: pikepdf.Dictionary) -> str:
            destination = self._path(f"extgstate_{name}.pdf")

            def add_state(pdf) -> None:
                page = pdf.pages[0]
                page.obj["/Resources"]["/ExtGState"] = pikepdf.Dictionary(
                    GScheck=state
                )
                page.obj["/Contents"] = pikepdf.Array(
                    [
                        pdf.make_stream(b"q /GScheck gs\n"),
                        page.obj["/Contents"],
                        pdf.make_stream(b"Q\n"),
                    ]
                )

            return self._rewrite_pdf(source, destination, add_state)

        opaque = pikepdf.Dictionary(
            Type=pikepdf.Name("/ExtGState"),
            BM=pikepdf.Name("/Normal"),
            CA=1,
            ca=1,
            SMask=pikepdf.Name("/None"),
        )
        self.assertEqual(scan_pdf(write_variant("opaque", opaque))[1][0].strategy, "native")
        close_pdf_cache()

        unsafe_states = {
            "blend": pikepdf.Dictionary(BM=pikepdf.Name("/Multiply")),
            "overprint": pikepdf.Dictionary(OP=True),
            "unknown": pikepdf.Dictionary(FL=1),
        }
        for name, state in unsafe_states.items():
            with self.subTest(name=name):
                self.assertEqual(
                    scan_pdf(write_variant(name, state))[1][0].strategy,
                    "render",
                )
                close_pdf_cache()

    def test_masks_special_colors_and_composition_force_render(self) -> None:
        source, _images = self._image_pdf(image_format="JPEG")

        def mutate_image(mutator):
            def apply(pdf) -> None:
                stream = next(iter(pdf.pages[0].images.values()))
                mutator(pdf, stream)

            return apply

        variants = {
            "mask": mutate_image(
                lambda _pdf, stream: stream.__setitem__(
                    "/Mask", pikepdf.Array([0, 0, 0, 0, 0, 0])
                )
            ),
            "decode": mutate_image(
                lambda _pdf, stream: stream.__setitem__(
                    "/Decode", pikepdf.Array([1, 0, 1, 0, 1, 0])
                )
            ),
            "icc": mutate_image(
                lambda pdf, stream: stream.__setitem__(
                    "/ColorSpace",
                    pikepdf.Array(
                        [
                            pikepdf.Name("/ICCBased"),
                            _icc_profile_stream(pdf),
                        ]
                    ),
                )
            ),
            "optional_content": mutate_image(
                lambda _pdf, stream: stream.__setitem__(
                    "/OC",
                    pikepdf.Dictionary(
                        Type=pikepdf.Name("/OCG"),
                        Name=pikepdf.String("Layer"),
                    ),
                )
            ),
            "render_intent": mutate_image(
                lambda _pdf, stream: stream.__setitem__(
                    "/Intent", pikepdf.Name("/Perceptual")
                )
            ),
            "unknown_image_key": mutate_image(
                lambda _pdf, stream: stream.__setitem__("/CustomEffect", True)
            ),
            "clip": lambda pdf: _prepend_content(
                pdf, b"q 0 0 10 10 re W n\n", b"Q\n"
            ),
            "text": lambda pdf: _prepend_content(pdf, b"BT ET\n", b""),
            "two_images": _append_duplicate_image,
            "page_group": lambda pdf: pdf.pages[0].obj.__setitem__(
                "/Group",
                pikepdf.Dictionary(S=pikepdf.Name("/Transparency")),
            ),
        }
        for name, mutator in variants.items():
            with self.subTest(name=name):
                destination = self._path(f"unsafe_{name}.pdf")
                self._rewrite_pdf(source, destination, mutator)
                self.assertEqual(scan_pdf(destination)[1][0].strategy, "render")
                close_pdf_cache()

        palette_path = self._path("palette.png")
        Image.new("P", (64, 48), 1).save(palette_path)
        palette_pdf = self._path("palette.pdf")
        Path(palette_pdf).write_bytes(img2pdf.convert(palette_path))
        self.assertEqual(scan_pdf(palette_pdf)[1][0].strategy, "render")

    def test_vector_paint_forces_render_without_dropping_page(self) -> None:
        source, _images = self._image_pdf(image_format="JPEG")
        altered = self._path("vector.pdf")

        def add_vector(pdf) -> None:
            page = pdf.pages[0]
            vector = pdf.make_stream(b"0 0 0 rg 0 0 3 3 re f\n")
            page.obj["/Contents"] = pikepdf.Array([page.obj["/Contents"], vector])

        self._rewrite_pdf(source, altered, add_vector)
        _identity, plans = scan_pdf(altered)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].strategy, "render")

    def test_blank_page_materializes_as_one_rgb_image(self) -> None:
        pdf_path = self._path("blank.pdf")
        with pikepdf.new() as pdf:
            pdf.add_blank_page(page_size=(72, 72))
            pdf.save(pdf_path)
        _identity, plans = scan_pdf(pdf_path)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].strategy, "render")
        output = self._path("blank.png")
        materialize_page(pdf_path, plans[0], output)
        with Image.open(output) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (600, 600))

    def test_zero_page_pdf_is_not_treated_as_readable(self) -> None:
        pdf_path = self._path("zero_pages.pdf")
        with pikepdf.new() as pdf:
            pdf.save(pdf_path)

        with self.assertRaises(PdfImportError) as raised:
            scan_pdf(pdf_path)

        self.assertEqual(raised.exception.code, "PDF_PAGE_PLAN_FAILED")
        self.assertIsNone(raised.exception.page_index)

    def test_cropbox_mismatch_forces_render(self) -> None:
        source, _images = self._image_pdf(image_format="JPEG")
        cropped = self._path("cropped.pdf")

        def crop(pdf) -> None:
            media = [float(value) for value in pdf.pages[0].obj["/MediaBox"]]
            media[2] -= 1
            pdf.pages[0].obj["/CropBox"] = pikepdf.Array(media)

        self._rewrite_pdf(source, cropped, crop)
        self.assertEqual(scan_pdf(cropped)[1][0].strategy, "render")

    def test_password_protected_pdf_reports_stable_error(self) -> None:
        source, _images = self._image_pdf(image_format="JPEG")
        encrypted = self._path("encrypted.pdf")
        with pikepdf.open(source) as pdf:
            pdf.save(
                encrypted,
                encryption=pikepdf.Encryption(owner="owner", user="reader", R=6),
            )

        with self.assertRaises(PdfImportError) as raised:
            scan_pdf(encrypted)
        self.assertEqual(raised.exception.code, "PDF_PASSWORD_REQUIRED")
        self.assertEqual(raised.exception.detail_code, "password_required")
        self.assertNotIn(encrypted, str(raised.exception))

    def test_malformed_pdf_reports_sanitized_plan_error(self) -> None:
        malformed = self._path("malformed.pdf")
        Path(malformed).write_bytes(b"%PDF-1.7\nnot a valid document")
        with self.assertRaises(PdfImportError) as raised:
            scan_pdf(malformed)
        self.assertEqual(raised.exception.code, "PDF_PAGE_PLAN_FAILED")
        self.assertEqual(raised.exception.detail_code, "page_scan_failed")
        self.assertNotIn(malformed, str(raised.exception))

    def test_file_page_object_form_and_user_unit_limits_are_typed(self) -> None:
        source, _images = self._image_pdf(image_format="JPEG")

        with mock.patch("modules.utils.pdf_pages.PDF_FILE_SIZE_LIMIT", 0):
            with self.assertRaises(PdfImportError) as raised:
                scan_pdf(source)
        self.assertEqual(raised.exception.detail_code, "file_size_limit")

        close_pdf_cache()
        with mock.patch("modules.utils.pdf_pages.PDF_PAGE_COUNT_LIMIT", 0):
            with self.assertRaises(PdfImportError) as raised:
                scan_pdf(source)
        self.assertEqual(raised.exception.detail_code, "page_count_limit")

        invalid_user_unit = self._path("invalid_user_unit.pdf")
        self._rewrite_pdf(
            source,
            invalid_user_unit,
            lambda pdf: pdf.pages[0].obj.__setitem__("/UserUnit", 0),
        )
        with self.assertRaises(PdfImportError) as raised:
            scan_pdf(invalid_user_unit)
        self.assertEqual(raised.exception.detail_code, "invalid_user_unit")

        duplicate = self._path("object_limit.pdf")
        self._rewrite_pdf(source, duplicate, _append_duplicate_image)
        with mock.patch("modules.utils.pdf_pages.PDF_OBJECT_COUNT_LIMIT", 1):
            with self.assertRaises(PdfImportError) as raised:
                scan_pdf(duplicate)
        self.assertEqual(raised.exception.detail_code, "object_count_limit")

        import pypdfium2.raw as pdfium_c

        form = SimpleNamespace(type=pdfium_c.FPDF_PAGEOBJ_FORM, level=14)
        page = SimpleNamespace(get_objects=lambda max_depth: [form])
        with mock.patch.object(pdfium_c, "FPDFFormObj_CountObjects", return_value=1):
            with self.assertRaises(PdfImportError) as raised:
                _page_objects(page)
        self.assertEqual(raised.exception.detail_code, "form_depth_limit")

    def test_engine_page_count_mismatch_is_typed(self) -> None:
        source, _images = self._image_pdf(image_format="JPEG")

        class MismatchedPdfiumDocument:
            def __init__(self, _path) -> None:
                pass

            def init_forms(self) -> None:
                pass

            def __len__(self) -> int:
                return 2

            def close(self) -> None:
                pass

        with mock.patch("pypdfium2.PdfDocument", MismatchedPdfiumDocument):
            with self.assertRaises(PdfImportError) as raised:
                scan_pdf(source)

        self.assertEqual(raised.exception.code, "PDF_PAGE_PLAN_FAILED")
        self.assertEqual(raised.exception.detail_code, "page_count_mismatch")

    def test_pdfium_form_initialization_failure_is_fail_closed(self) -> None:
        source, _images = self._image_pdf(image_format="JPEG")

        class FormInitFailureDocument:
            def __init__(self, _path) -> None:
                pass

            def init_forms(self) -> None:
                raise RuntimeError("private parser detail")

            def close(self) -> None:
                pass

        with mock.patch("pypdfium2.PdfDocument", FormInitFailureDocument):
            with self.assertRaises(PdfImportError) as raised:
                scan_pdf(source)

        self.assertEqual(raised.exception.code, "PDF_PAGE_PLAN_FAILED")
        self.assertEqual(raised.exception.detail_code, "page_scan_failed")
        self.assertNotIn("private parser detail", str(raised.exception))

    def test_changed_source_invalidates_existing_plan(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        _identity, plans = scan_pdf(pdf_path)
        close_pdf_cache(pdf_path)
        with open(pdf_path, "ab") as output:
            output.write(b"\n")

        with self.assertRaises(PdfImportError) as raised:
            materialize_page(pdf_path, plans[0], self._path("changed.jpg"))
        self.assertEqual(raised.exception.code, "PDF_SOURCE_CHANGED")

    def test_plan_schema_mismatch_is_rejected(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        plan = replace(scan_pdf(pdf_path)[1][0], schema_version=0)

        with self.assertRaises(PdfImportError) as raised:
            materialize_page(pdf_path, plan, self._path("schema.jpg"))

        self.assertEqual(raised.exception.code, "PDF_SOURCE_CHANGED")
        self.assertEqual(raised.exception.detail_code, "plan_schema_mismatch")

    def test_modified_plan_is_rejected_against_canonical_scan(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        plan = scan_pdf(pdf_path)[1][0]
        modified = replace(plan, encoded_sha256="0" * 64)

        with self.assertRaises(PdfImportError) as raised:
            materialize_page(pdf_path, modified, self._path("modified.jpg"))

        self.assertEqual(raised.exception.code, "PDF_SOURCE_CHANGED")
        self.assertEqual(raised.exception.detail_code, "plan_source_mismatch")

    def test_transaction_publish_failure_removes_only_new_outputs(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG", count=3)
        _identity, plans = scan_pdf(pdf_path)
        outputs = [self._path(f"out_{index}.jpg") for index in range(3)]
        real_replace = os.replace
        publish_calls = 0

        def fail_second_publish(source: str, destination: str) -> None:
            nonlocal publish_calls
            if ".pdf_staging_" in source:
                publish_calls += 1
                if publish_calls == 2:
                    raise OSError("injected publish failure")
            real_replace(source, destination)

        with mock.patch(
            "modules.utils.pdf_pages.os.replace", side_effect=fail_second_publish
        ):
            with self.assertRaises(PdfImportError) as raised:
                materialize_transaction(pdf_path, list(zip(plans, outputs)))

        self.assertEqual(raised.exception.detail_code, "publish_failed")
        self.assertTrue(all(not os.path.exists(path) for path in outputs))
        self.assertTrue(os.path.isfile(pdf_path))

    def test_transaction_rejects_modified_nonfirst_plan(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG", count=2)
        plans = scan_pdf(pdf_path)[1]
        modified = replace(plans[1], encoded_sha256="0" * 64)
        outputs = [self._path("first.jpg"), self._path("second.jpg")]

        with self.assertRaises(PdfImportError) as raised:
            materialize_transaction(pdf_path, [(plans[0], outputs[0]), (modified, outputs[1])])

        self.assertEqual(raised.exception.code, "PDF_SOURCE_CHANGED")
        self.assertTrue(all(not os.path.exists(path) for path in outputs))

    def test_transaction_does_not_reject_from_estimated_disk_space(self) -> None:
        pdf_path, images = self._image_pdf(image_format="JPEG")
        _identity, plans = scan_pdf(pdf_path)
        output = self._path("materialized.jpg")
        with mock.patch(
            "modules.utils.pdf_pages.shutil.disk_usage",
            side_effect=AssertionError("transaction must not estimate free disk space"),
        ):
            materialize_transaction(pdf_path, [(plans[0], output)])

        self.assertEqual(Path(output).read_bytes(), Path(images[0]).read_bytes())

    def test_disk_space_error_classifier_handles_nested_platform_codes(self) -> None:
        quota_error = OSError(getattr(errno, "EDQUOT", errno.ENOSPC), "quota exceeded")
        wrapped = PdfImportError(
            "PDF_PAGE_MATERIALIZATION_FAILED", detail_code="native_validation_failed"
        )
        wrapped.__cause__ = quota_error
        self.assertTrue(pdf_pages_module._is_disk_space_error(wrapped))

        for winerror in (39, 112):
            with self.subTest(winerror=winerror):
                windows_error = OSError("disk full")
                windows_error.winerror = winerror
                self.assertTrue(pdf_pages_module._is_disk_space_error(windows_error))

        self.assertFalse(
            pdf_pages_module._is_disk_space_error(
                OSError(errno.EACCES, "permission denied")
            )
        )

    def test_transaction_actual_disk_full_preserves_source_and_destination(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        plan = scan_pdf(pdf_path)[1][0]
        output = self._path("disk_full.jpg")
        real_open = open

        def fail_staging_write(path, mode="r", *args, **kwargs):
            if ".pdf_staging_" in os.fspath(path) and "w" in mode:
                raise OSError(errno.ENOSPC, "injected disk full")
            return real_open(path, mode, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fail_staging_write):
            with self.assertRaises(PdfImportError) as raised:
                materialize_transaction(pdf_path, [(plan, output)])

        self.assertEqual(raised.exception.code, "PDF_DISK_SPACE_INSUFFICIENT")
        self.assertEqual(raised.exception.detail_code, "disk_space_insufficient")
        self.assertTrue(raised.exception.retryable)
        self.assertTrue(os.path.isfile(pdf_path))
        self.assertFalse(os.path.exists(output))
        self.assertFalse(any(Path(self.temp_dir).glob(".pdf_staging_*")))

    def test_transaction_disk_full_while_creating_staging_is_typed(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        plan = scan_pdf(pdf_path)[1][0]
        output = self._path("staging_disk_full.jpg")

        with mock.patch(
            "modules.utils.pdf_pages.tempfile.mkdtemp",
            side_effect=OSError(errno.ENOSPC, "injected disk full"),
        ):
            with self.assertRaises(PdfImportError) as raised:
                materialize_transaction(pdf_path, [(plan, output)])

        self.assertEqual(raised.exception.code, "PDF_DISK_SPACE_INSUFFICIENT")
        self.assertEqual(raised.exception.detail_code, "disk_space_insufficient")
        self.assertTrue(raised.exception.retryable)
        self.assertTrue(os.path.isfile(pdf_path))
        self.assertFalse(os.path.exists(output))

    def test_transaction_other_staging_error_is_sanitized(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        plan = scan_pdf(pdf_path)[1][0]
        output = self._path("staging_denied.jpg")

        with mock.patch(
            "modules.utils.pdf_pages.tempfile.mkdtemp",
            side_effect=OSError(errno.EACCES, "injected permission denied"),
        ):
            with self.assertRaises(PdfImportError) as raised:
                materialize_transaction(pdf_path, [(plan, output)])

        self.assertEqual(raised.exception.code, "PDF_PAGE_MATERIALIZATION_FAILED")
        self.assertEqual(raised.exception.detail_code, "publish_failed")
        self.assertFalse(raised.exception.retryable)
        self.assertTrue(os.path.isfile(pdf_path))
        self.assertFalse(os.path.exists(output))

    def test_transaction_disk_full_during_publish_rolls_back_outputs(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG", count=2)
        plans = scan_pdf(pdf_path)[1]
        outputs = [self._path(f"disk_full_{index}.jpg") for index in range(2)]
        real_replace = os.replace
        publish_calls = 0

        def fail_second_publish(source: str, destination: str) -> None:
            nonlocal publish_calls
            if ".pdf_staging_" in source:
                publish_calls += 1
                if publish_calls == 2:
                    raise OSError(errno.ENOSPC, "injected disk full")
            real_replace(source, destination)

        with mock.patch("modules.utils.pdf_pages.os.replace", side_effect=fail_second_publish):
            with self.assertRaises(PdfImportError) as raised:
                materialize_transaction(pdf_path, list(zip(plans, outputs)))

        self.assertEqual(raised.exception.code, "PDF_DISK_SPACE_INSUFFICIENT")
        self.assertEqual(raised.exception.detail_code, "disk_space_insufficient")
        self.assertTrue(raised.exception.retryable)
        self.assertTrue(os.path.isfile(pdf_path))
        self.assertTrue(all(not os.path.exists(path) for path in outputs))
        self.assertFalse(any(Path(self.temp_dir).glob(".pdf_staging_*")))

    def test_transaction_never_reuses_abandoned_staging_content(self) -> None:
        pdf_path, images = self._image_pdf(image_format="JPEG")
        plan = scan_pdf(pdf_path)[1][0]
        abandoned = self._path(".pdf_staging_abandoned")
        os.makedirs(abandoned)
        Path(abandoned, "000001.jpg").write_bytes(b"untrusted stale bytes")
        output = self._path("fresh.jpg")

        materialize_transaction(pdf_path, [(plan, output)])

        self.assertEqual(Path(output).read_bytes(), Path(images[0]).read_bytes())
        self.assertEqual(
            Path(abandoned, "000001.jpg").read_bytes(), b"untrusted stale bytes"
        )

    def test_transaction_preserves_cancellation_type_and_cleans_outputs(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG", count=2)
        _identity, plans = scan_pdf(pdf_path)
        outputs = [self._path(f"cancel_{index}.jpg") for index in range(2)]
        checks = 0

        def should_cancel() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 4

        with self.assertRaises(OperationCancelledError):
            materialize_transaction(
                pdf_path,
                list(zip(plans, outputs)),
                should_cancel=should_cancel,
            )

        self.assertTrue(all(not os.path.exists(path) for path in outputs))

    def test_file_handler_preflight_keeps_unopened_pdf_pages_lazy(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG", count=3)
        handler = FileHandler()
        prepared = handler.prepare_files([pdf_path])

        self.assertEqual(len(prepared), 3)
        self.assertTrue(os.path.isfile(prepared[0]))
        self.assertFalse(os.path.exists(prepared[1]))
        self.assertFalse(os.path.exists(prepared[2]))

        self.assertEqual(handler.preflight_for_processing(prepared), 3)
        self.assertTrue(os.path.isfile(prepared[0]))
        self.assertFalse(os.path.exists(prepared[1]))
        self.assertFalse(os.path.exists(prepared[2]))
        resource_plan = handler.image_resource_plan()
        self.assertEqual(resource_plan.page_count, 3)
        self.assertTrue(resource_plan.hard_cap_passed)

        entries = list_archive_image_entries(pdf_path)
        self.assertEqual([entry["page_index"] for entry in entries], [0, 1, 2])
        self.assertEqual([entry["ext"] for entry in entries], [".jpg"] * 3)

    def test_preflight_rechecks_source_even_when_first_page_is_ready(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        handler = FileHandler()
        prepared = handler.prepare_files([pdf_path])
        self.assertTrue(os.path.isfile(prepared[0]))

        close_pdf_cache(pdf_path)
        with open(pdf_path, "ab") as output:
            output.write(b"\n")

        with self.assertRaises(PdfImportError) as raised:
            handler.preflight_for_processing(prepared)

        self.assertEqual(raised.exception.code, "PDF_SOURCE_CHANGED")

    def test_file_handler_preflight_never_materializes_multiple_pdf_groups(self) -> None:
        first_pdf, _images = self._image_pdf(image_format="JPEG", count=2)
        first_pdf_copy = self._path("first.pdf")
        os.replace(first_pdf, first_pdf_copy)
        second_pdf, _images = self._image_pdf(image_format="JPEG", count=2)
        second_pdf_copy = self._path("second.pdf")
        os.replace(second_pdf, second_pdf_copy)
        handler = FileHandler()
        prepared = handler.prepare_files([first_pdf_copy, second_pdf_copy])
        first_lazy_page_two = prepared[1]
        second_lazy_page_two = prepared[3]
        self.assertEqual(handler.preflight_for_processing(prepared), 4)

        self.assertFalse(os.path.exists(first_lazy_page_two))
        self.assertFalse(os.path.exists(second_lazy_page_two))
        self.assertTrue(os.path.isfile(prepared[0]))
        self.assertTrue(os.path.isfile(prepared[2]))

    def test_pdf_header_preflight_does_not_validate_a_missing_materialized_page(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG", count=2)
        handler = FileHandler()
        prepared = handler.prepare_files([pdf_path])
        self.assertFalse(os.path.exists(prepared[1]))

        with mock.patch(
            "modules.utils.file_handler.validate_materialized_page",
            side_effect=AssertionError("preflight must not inspect a rendered page"),
        ):
            self.assertEqual(handler.preflight_for_processing([prepared[1]]), 1)

        self.assertFalse(os.path.exists(prepared[1]))

    def test_pdf_warnings_are_filtered_to_selected_paths(self) -> None:
        regular_pdf, _images = self._image_pdf(image_format="JPEG")
        regular_copy = self._path("regular.pdf")
        os.replace(regular_pdf, regular_copy)

        wide_image = self._path("selected_wide.jpg")
        Image.new("RGB", (25_001, 26), (20, 40, 60)).save(wide_image, "JPEG")
        wide_pdf = self._path("selected_wide.pdf")
        Path(wide_pdf).write_bytes(
            img2pdf.convert(
                wide_image,
                layout_fun=img2pdf.get_fixed_dpi_layout_fun((600, 600)),
            )
        )

        handler = FileHandler()
        prepared = handler.prepare_files([regular_copy, wide_pdf])

        self.assertEqual(handler.get_pdf_import_warnings([prepared[0]]), [])
        selected_warnings = handler.get_pdf_import_warnings([prepared[1]])
        self.assertEqual(len(selected_warnings), 1)
        self.assertEqual(selected_warnings[0]["page_number"], 1)

    def test_concurrent_transactions_publish_one_valid_page(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        _identity, plans = scan_pdf(pdf_path)
        output = self._path("concurrent.jpg")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(materialize_transaction, pdf_path, [(plans[0], output)])
                for _ in range(2)
            ]
            for future in futures:
                future.result()

        self.assertTrue(validate_materialized_page(plans[0], output))

    def test_cache_is_limited_to_four_documents_and_close_releases_source(self) -> None:
        first_path = ""
        for index in range(5):
            image_path = self._path(f"cache_{index}.jpg")
            pdf_path = self._path(f"cache_{index}.pdf")
            Image.new("RGB", (16, 16), (index, index, index)).save(image_path, "JPEG")
            Path(pdf_path).write_bytes(img2pdf.convert(image_path))
            scan_pdf(pdf_path)
            if index == 0:
                first_path = os.path.abspath(pdf_path)

        self.assertEqual(len(_PDF_CACHE), 4)
        self.assertNotIn(first_path, {path for _schema, path in _PDF_CACHE})

        last_path = self._path("cache_4.pdf")
        close_pdf_cache(last_path)
        renamed = self._path("cache_4_renamed.pdf")
        os.replace(last_path, renamed)
        self.assertTrue(os.path.isfile(renamed))

    def test_cache_close_waits_for_active_scan_and_prevents_reinsertion(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        entered = threading.Event()
        release = threading.Event()
        real_scan_page = pdf_pages_module._scan_page

        def slow_scan_page(entry, page_index):
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            return real_scan_page(entry, page_index)

        with mock.patch(
            "modules.utils.pdf_pages._scan_page", side_effect=slow_scan_page
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                scan_future = executor.submit(scan_pdf, pdf_path)
                self.assertTrue(entered.wait(timeout=5))
                close_future = executor.submit(close_pdf_cache)
                time.sleep(0.05)
                self.assertFalse(close_future.done())
                release.set()
                scan_future.result(timeout=5)
                close_future.result(timeout=5)

        self.assertEqual(len(_PDF_CACHE), 0)
        renamed = self._path("after_close.pdf")
        os.replace(pdf_path, renamed)
        self.assertTrue(os.path.isfile(renamed))

    def test_materialize_hashes_source_once_not_twice(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        plan = scan_pdf(pdf_path)[1][0]

        with mock.patch(
            "modules.utils.pdf_pages._source_identity",
            wraps=pdf_pages_module._source_identity,
        ) as identity:
            materialize_page(pdf_path, plan, self._path("single_hash.jpg"))

        self.assertEqual(identity.call_count, 1)

    def test_render_limits_preserve_aspect_ratio_and_bound_dimensions(self) -> None:
        requested, applied, applied_dpi, capped = _render_dimensions(
            (10_000.0, 8_000.0), 1.0, 600.0
        )
        self.assertTrue(capped)
        self.assertGreater(requested[0] * requested[1], PDF_RENDER_PIXEL_LIMIT)
        self.assertLessEqual(applied[0] * applied[1], PDF_RENDER_PIXEL_LIMIT)
        self.assertLessEqual(max(applied), PDF_RENDER_LONG_SIDE_LIMIT)
        self.assertLess(applied_dpi, 600.0)
        self.assertAlmostEqual(applied[0] / applied[1], requested[0] / requested[1], places=3)

    def test_cap_warning_payload_is_sanitized_and_one_based(self) -> None:
        pdf_path, _images = self._image_pdf(image_format="JPEG")
        plan = replace(
            scan_pdf(pdf_path)[1][0],
            page_index=2,
            cap_applied=True,
            requested_size=(30_000, 20_000),
            render_size=(12_247, 8_165),
        )

        self.assertEqual(
            plan.warning_payload(),
            {
                "code": "PDF_RESOURCE_LIMIT",
                "page_number": 3,
                "requested_width": 30_000,
                "requested_height": 20_000,
                "applied_width": 12_247,
                "applied_height": 8_165,
            },
        )

    def test_native_candidate_above_long_side_cap_becomes_render_plan(self) -> None:
        image_path = self._path("wide.jpg")
        Image.new("RGB", (25_001, 26), (20, 40, 60)).save(image_path, "JPEG")
        pdf_path = self._path("wide.pdf")
        Path(pdf_path).write_bytes(
            img2pdf.convert(
                image_path,
                layout_fun=img2pdf.get_fixed_dpi_layout_fun((600, 600)),
            )
        )

        with mock.patch(
            "modules.utils.pdf_pages._native_candidate",
            side_effect=AssertionError("capped native input must not be decoded during scan"),
        ):
            plan = scan_pdf(pdf_path)[1][0]

        self.assertEqual(plan.strategy, "render")
        self.assertTrue(plan.cap_applied)
        self.assertGreater(plan.requested_size[0], PDF_RENDER_LONG_SIDE_LIMIT)
        self.assertLessEqual(plan.render_size[0], PDF_RENDER_LONG_SIDE_LIMIT)
        self.assertEqual(plan.warning_payload()["page_number"], 1)

    def test_render_limit_boundaries_are_inclusive(self) -> None:
        exact_pixels = _render_dimensions((1200.0, 1200.0), 1.0, 600.0)
        over_pixels = _render_dimensions((1200.12, 1200.0), 1.0, 600.0)
        exact_side = _render_dimensions((3000.0, 120.0), 1.0, 600.0)
        over_side = _render_dimensions((3000.12, 120.0), 1.0, 600.0)

        self.assertEqual(exact_pixels[0], (10_000, 10_000))
        self.assertFalse(exact_pixels[3])
        self.assertTrue(over_pixels[3])
        self.assertEqual(exact_side[0], (25_000, 1_001))
        self.assertFalse(exact_side[3])
        self.assertTrue(over_side[3])

    def test_render_fallback_is_serial_across_documents(self) -> None:
        paths = []
        plans = []
        for index in range(2):
            path = self._path(f"blank_{index}.pdf")
            with pikepdf.new() as pdf:
                pdf.add_blank_page(page_size=(72, 72))
                pdf.save(path)
            paths.append(path)
            plans.append(scan_pdf(path)[1][0])

        class ProbeSemaphore:
            def __init__(self) -> None:
                self._semaphore = threading.Semaphore(1)
                self._lock = threading.Lock()
                self.active = 0
                self.peak = 0

            def __enter__(self):
                self._semaphore.acquire()
                with self._lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.03)
                return self

            def __exit__(self, *_args):
                with self._lock:
                    self.active -= 1
                self._semaphore.release()

        probe = ProbeSemaphore()
        with mock.patch("modules.utils.pdf_pages._RENDER_SEMAPHORE", probe):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        materialize_page,
                        path,
                        plan,
                        self._path(f"rendered_{index}.png"),
                    )
                    for index, (path, plan) in enumerate(zip(paths, plans))
                ]
                for future in futures:
                    future.result()

        self.assertEqual(probe.peak, 1)

    def test_cancel_while_waiting_for_render_prevents_render_start(self) -> None:
        pdf_path = self._path("cancel_render.pdf")
        with pikepdf.new() as pdf:
            pdf.add_blank_page(page_size=(72, 72))
            pdf.save(pdf_path)
        plan = scan_pdf(pdf_path)[1][0]
        output = self._path("cancel_render.png")

        class CancellationGate:
            entered = False

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, *_args):
                return False

        gate = CancellationGate()
        with mock.patch("modules.utils.pdf_pages._RENDER_SEMAPHORE", gate):
            with self.assertRaises(OperationCancelledError):
                materialize_transaction(
                    pdf_path,
                    [(plan, output)],
                    should_cancel=lambda: gate.entered,
                )

        self.assertTrue(gate.entered)
        self.assertFalse(os.path.exists(output))

    def test_effective_dpi_uses_user_unit_and_five_percent_area_boundary(self) -> None:
        def image(bounds, matrix, pixels):
            return SimpleNamespace(
                get_bounds=lambda: bounds,
                get_matrix=lambda: SimpleNamespace(get=lambda: matrix),
                get_px_size=lambda: pixels,
            )

        page = (0.0, 0.0, 100.0, 100.0)
        below_threshold = image((0.0, 0.0, 49.0, 10.0), (10, 0, 0, 10, 0, 0), (1000, 1000))
        at_threshold = image((0.0, 0.0, 50.0, 10.0), (10, 0, 0, 10, 0, 0), (1000, 1000))

        self.assertEqual(_requested_dpi([below_threshold], page, 2.0), 600.0)
        self.assertEqual(_requested_dpi([at_threshold], page, 2.0), 3600.0)


if __name__ == "__main__":
    unittest.main()
