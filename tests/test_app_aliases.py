import pytest

from vlaclaw.backends.adb import AdbBackend
from vlaclaw.backends.app_aliases import is_android_package_name, normalize_adb_app_identifier
from vlaclaw.cli import _coerce_app_aliases


def test_display_names_resolve_chinese_and_english() -> None:
    cases = {
        "微信": "com.tencent.mm",
        "WeChat": "com.tencent.mm",
        "Settings": "com.android.settings",
        "settings": "com.android.settings",
        "美团": "com.sankuai.meituan",
        "Meituan": "com.sankuai.meituan",
        "红果": "com.phoenix.read",
        "红果免费短剧": "com.phoenix.read",
        "喜马拉雅": "com.ximalaya.ting.android",
        "Himalaya": "com.ximalaya.ting.android",
        "Chrome": "com.android.chrome",
        "支付宝": "com.eg.android.AlipayGphone",
        "Alipay": "com.eg.android.AlipayGphone",
    }
    for alias, package in cases.items():
        assert normalize_adb_app_identifier(alias) == package


def test_extra_aliases_not_in_display_names() -> None:
    cases = {
        "weixin": "com.tencent.mm",
        "B站": "tv.danmaku.bili",
        "小破站": "tv.danmaku.bili",
        "油管": "com.google.android.youtube",
        "yt": "com.google.android.youtube",
        "chrome browser": "com.android.chrome",
        "himalaya": "com.ximalaya.ting.android",
        "maps": "com.google.android.apps.maps",
        "sms": "com.google.android.apps.messaging",
        "deskclock": "com.google.android.deskclock",
        "android settings": "com.android.settings",
    }
    for alias, package in cases.items():
        assert normalize_adb_app_identifier(alias) == package


def test_prefix_and_suffix_stripping() -> None:
    cases = {
        "打开微信": "com.tencent.mm",
        "微信App": "com.tencent.mm",
        "打开喜马拉雅App": "com.ximalaya.ting.android",
        "open ximalaya app": "com.ximalaya.ting.android",
        "打开红果免费短剧": "com.phoenix.read",
        "launch Settings": "com.android.settings",
        "小红书客户端": "com.xingin.xhs",
    }
    for alias, package in cases.items():
        assert normalize_adb_app_identifier(alias) == package


def test_existing_packages_are_preserved() -> None:
    assert normalize_adb_app_identifier("com.tencent.mm") == "com.tencent.mm"
    assert normalize_adb_app_identifier("com.eg.android.AlipayGphone") == (
        "com.eg.android.AlipayGphone"
    )
    assert normalize_adb_app_identifier("com.taobao.taobao") == "com.taobao.taobao"
    assert normalize_adb_app_identifier("com.google.android.gm") == "com.google.android.gm"


def test_mobileworld_synthetic_packages_are_not_used() -> None:
    assert normalize_adb_app_identifier("gmail") == "com.google.android.gm"
    assert normalize_adb_app_identifier("Gmail") == "com.google.android.gm"
    assert normalize_adb_app_identifier("calendar") == "com.android.calendar"
    assert normalize_adb_app_identifier("taobao") == "com.taobao.taobao"
    assert normalize_adb_app_identifier("gallery") == "com.android.gallery3d"
    assert normalize_adb_app_identifier("mail") == "unknown"


def test_adb_overrides_real_device_packages() -> None:
    assert normalize_adb_app_identifier("Mastodon") == "org.joinmastodon.android"
    assert normalize_adb_app_identifier("Mastodon App") == "org.joinmastodon.android"
    assert normalize_adb_app_identifier("org.joinmastodon.android.mastodon") == (
        "org.joinmastodon.android"
    )


def test_extra_aliases_override_builtins() -> None:
    extra = {"微信": "com.example.wechat", "mybank": "com.foo.bank"}
    assert normalize_adb_app_identifier("微信", extra_aliases=extra) == "com.example.wechat"
    assert normalize_adb_app_identifier("打开mybank", extra_aliases=extra) == "com.foo.bank"
    assert normalize_adb_app_identifier("Settings", extra_aliases=extra) == "com.android.settings"


def test_unknown_names_return_unknown() -> None:
    assert normalize_adb_app_identifier("") == "unknown"
    assert normalize_adb_app_identifier("not-an-app") == "unknown"
    assert not is_android_package_name("unknown")
    assert not is_android_package_name("微信")
    assert is_android_package_name("com.tencent.mm")


def test_backend_resolves_open_and_close_app_names() -> None:
    backend = AdbBackend(app_aliases={"mybank": "com.foo.bank"})
    assert backend._resolve_app_package("微信", action_type="open_app") == "com.tencent.mm"
    assert backend._resolve_app_package("mybank", action_type="close_app") == "com.foo.bank"
    with pytest.raises(ValueError, match="cannot resolve"):
        backend._resolve_app_package("not-an-app", action_type="open_app")


def test_coerce_app_aliases_from_config() -> None:
    assert _coerce_app_aliases(None) == {}
    assert _coerce_app_aliases({" 微信 ": " com.tencent.mm "}) == {"微信": "com.tencent.mm"}
    with pytest.raises(ValueError, match="must be a mapping"):
        _coerce_app_aliases(["微信"])
    with pytest.raises(ValueError, match="Invalid adb.app_aliases"):
        _coerce_app_aliases({"微信": ""})
