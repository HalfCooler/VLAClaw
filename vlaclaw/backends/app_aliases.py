"""Android app display-name and alias lookup for ADB launch/stop.

Tables are ported from GUIClaw ``skills/normalization.py``. MobileWorld
synthetic remaps (``com.gmailclone``, Fossify Calendar, etc.) are omitted:
``monkey -p`` / ``am force-stop`` need packages installed on the device.
Override or extend via ``adb.app_aliases`` in ``~/.vlaclaw/config.yaml``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_ANDROID_PACKAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+$")

# ---------------------------------------------------------------------------
# Package name -> display name (source of reverse aliases)
# ---------------------------------------------------------------------------

_ANDROID_PACKAGE_DISPLAY_NAMES: dict[str, str] = {
    # Social & Communication
    "com.tencent.mm": "微信/WeChat",
    "com.tencent.mobileqq": "QQ",
    "com.sina.weibo": "微博/Weibo",
    "com.zhihu.android": "知乎/Zhihu",
    "com.xingin.xhs": "小红书/RedNote",
    "com.twitter.android": "X/Twitter",
    "com.whatsapp": "WhatsApp",
    "org.telegram.messenger": "Telegram",
    "com.facebook.katana": "Facebook",
    "tv.danmaku.bili": "哔哩哔哩/Bilibili",
    # Shopping & Food
    "com.taobao.taobao": "淘宝/Taobao",
    "com.jingdong.app.mall": "京东/JD",
    "com.xunmeng.pinduoduo": "拼多多/Pinduoduo",
    "com.taobao.idlefish": "闲鱼/Xianyu",
    "com.sankuai.meituan": "美团/Meituan",
    "com.dianping.v1": "大众点评/Dianping",
    "com.pupumall.customer": "朴朴超市/PuPu",
    "cn.walmart.app": "沃尔玛/Walmart",
    "com.lucky.luckyclient": "瑞幸咖啡/Luckin",
    "com.yek.android.kfc.activitys": "肯德基/KFC",
    # Transport & Maps
    "com.sdu.didi.psnger": "滴滴出行/DiDi",
    "com.autonavi.minimap": "高德地图/Amap",
    "com.baidu.BaiduMap": "百度地图/Baidu Maps",
    "com.tencent.map": "腾讯地图/Tencent Maps",
    "cn.caocaokeji.user": "曹操出行/CaoCao",
    "com.lalamove.huolala.client": "货拉拉/Huolala",
    "com.jingyao.easybike": "哈啰/Hellobike",
    "com.ygkj.chelaile.standard": "车来了/Chelaile",
    # Travel
    "ctrip.android.view": "携程/Ctrip",
    "cn.damai": "大麦/Damai",
    "com.csair.mbp": "南方航空/CSAir",
    "com.rytong.ceair": "东方航空/CEAir",
    "com.umetrip.android.msky.app": "航旅纵横/Umetrip",
    "com.MobileTicket": "12306",
    # Finance
    "com.eg.android.AlipayGphone": "支付宝/Alipay",
    "com.icbc": "工商银行/ICBC",
    "com.icbc.elife": "工银e生活",
    "com.unionpay": "云闪付/UnionPay",
    "com.finshell.wallet": "数字人民币/e-CNY",
    "com.pingan.paces.ccms": "平安口袋银行/PingAn",
    "com.chinamworld.bocmbci": "中国银行/BOC",
    "com.bochk.app.aos": "中银香港/BOCHK",
    "com.android.bankabc": "农业银行/ABC",
    "cn.gov.tax.its": "个人所得税/ITS",
    "com.usmart.stock": "uSMART HK",
    "com.usmart.sg.stock": "uSMART SG",
    # Entertainment
    "com.ss.android.ugc.aweme": "抖音/Douyin",
    "com.netease.cloudmusic": "网易云音乐/NetEase Music",
    "com.ximalaya.ting.android": "喜马拉雅/Ximalaya/Himalaya",
    "com.tencent.qqmusic": "QQ音乐/QQ Music",
    "com.kugou.android": "酷狗音乐/Kugou Music",
    "com.tencent.karaoke": "全民K歌/WeSing",
    "com.qiyi.video": "爱奇艺/iQIYI",
    "com.tencent.qqlive": "腾讯视频/Tencent Video",
    "com.youku.phone": "优酷/Youku",
    "com.phoenix.read": "红果免费短剧/红果/Hongguo",
    "com.smile.gifmaker": "快手/Kuaishou",
    "com.kuaishou.nebula": "快手极速版/Kuaishou Lite",
    "com.ss.android.article.news": "今日头条/Toutiao",
    "com.google.android.youtube": "YouTube",
    "com.bytedance.dreamina": "即梦/Dreamina",
    "com.hunantv.imgo.activity": "芒果TV/Mango TV",
    # Work & Productivity
    "com.ss.android.lark": "飞书/Lark",
    "com.alibaba.android.rimet": "钉钉/DingTalk",
    "com.tencent.wework": "企业微信/WeCom",
    "com.tencent.wemeet.app": "腾讯会议/VooV",
    "com.tencent.docs": "腾讯文档/Tencent Docs",
    "com.tencent.androidqqmail": "QQ邮箱/QQ Mail",
    "cn.wps.moffice_eng": "WPS Office",
    "com.microsoft.office.outlook": "Outlook",
    "com.microsoft.skydrive": "OneDrive",
    "com.microsoft.office.officehub": "Microsoft Office",
    "notion.id": "Notion",
    "md.obsidian": "Obsidian",
    # AI
    "com.deepseek.chat": "DeepSeek",
    "com.openai.chatgpt": "ChatGPT",
    "com.aliyun.tongyi": "通义千问/Tongyi",
    "ai.x.grok": "Grok",
    "com.google.android.apps.bard": "Gemini",
    "com.tencent.hunyuan.app.chat": "腾讯混元/Hunyuan",
    "com.pocketpalai": "PocketPal AI",
    # Reading & Cloud
    "com.tencent.weread": "微信读书/WeRead",
    "com.dragon.read": "番茄小说/Fanqie Novel",
    "com.qidian.QDReader": "起点读书/Qidian",
    "com.baidu.searchbox": "百度/Baidu",
    "com.baidu.netdisk": "百度网盘/Baidu Netdisk",
    "net.csdn.csdnplus": "CSDN",
    # Google
    "com.android.chrome": "Chrome",
    "com.google.android.gm": "Gmail",
    "com.google.android.apps.maps": "Google Maps",
    "com.google.android.apps.photos": "Google Photos",
    "com.google.android.apps.docs": "Google Docs",
    "com.google.android.apps.messaging": "Google Messages",
    "com.google.android.calendar": "Google Calendar",
    "com.google.android.googlequicksearchbox": "Google",
    "com.google.android.contacts": "Google Contacts",
    "com.google.android.dialer": "Google Phone",
    "com.google.android.apps.labs.language.tailwind": "NotebookLM",
    # System
    "com.android.settings": "Settings",
    "com.android.contacts": "Contacts",
    "com.android.dialer": "Phone",
    "com.android.mms": "Messages",
    "com.android.camera2": "Camera",
    "com.android.gallery3d": "Gallery",
    "com.android.calculator2": "Calculator",
    "com.android.calendar": "Calendar",
    "com.android.deskclock": "Clock",
    "com.android.documentsui": "Files",
    "com.android.vending": "Play Store",
    "com.android.email": "Email",
    # OPPO/ColorOS System
    "com.coloros.soundrecorder": "录音/Sound Recorder",
    "com.coloros.filemanager": "文件管理/File Manager",
    "com.coloros.weather2": "天气/Weather",
    "com.coloros.calendar": "日历/Calendar",
    "com.coloros.calculator": "计算器/Calculator",
    "com.coloros.compass2": "指南针/Compass",
    "com.coloros.alarmclock": "闹钟/Alarm Clock",
    "com.coloros.note": "备忘录/Notes",
    "com.coloros.translate": "翻译/Translate",
    "com.coloros.backuprestore": "备份与恢复/Backup",
    "com.coloros.gallery3d": "相册/Gallery",
    "com.coloros.camera": "相机/Camera",
    "com.coloros.phonemanager": "手机管家/Phone Manager",
    "com.coloros.safecenter": "安全中心/Security Center",
    "com.coloros.oshare": "互传/OShare",
    "com.heytap.browser": "浏览器/Browser",
    "com.heytap.music": "音乐/Music",
    "com.heytap.themestore": "主题商店/Theme Store",
    "com.nearme.gamecenter": "游戏中心/Game Center",
    "com.oppo.market": "应用商店/App Store",
    "com.oppo.quicksearchbox": "搜索/Search",
    # Developer & Tools
    "com.github.android": "GitHub",
    "org.zotero.android": "Zotero",
    "com.server.auditor.ssh.client": "Termius",
    "com.quark.browser": "夸克/Quark",
    "com.tencent.mtt": "QQ浏览器/QQ Browser",
    "com.UCMobile": "UC浏览器/UC Browser",
    "mark.via": "Via Browser",
    # VPN & Security
    "com.tailscale.ipn": "Tailscale",
    "com.github.metacubex.clash.meta": "Clash Meta",
    "com.oray.sunlogin": "向日葵/Sunlogin",
    "com.sangfor.atrust": "aTrust",
    "com.azure.authenticator": "MS Authenticator",
    "com.duosecurity.duomobile": "Duo Mobile",
    # Telecom
    "com.ct.client": "中国电信/China Telecom",
    "com.greenpoint.android.mc10086.activity": "中国移动/China Mobile",
    "com.redteamobile.roaming": "红茶移动/RedTea",
    # Transit
    "com.szt.pay": "深圳通/SZT",
    "com.lingnanpass": "岭南通/Lingnan Pass",
    # Health
    "com.huawei.health": "华为健康/Huawei Health",
    "com.mi.health": "小米健康/Mi Health",
    "com.leoao.fitness": "乐刻运动/Leoao",
    # Gaming
    "com.valvesoftware.android.steam.community": "Steam",
    "com.megacrit.cardcrawl": "Slay the Spire",
    "com.playstack.balatro.android": "Balatro",
    "com.scee.psxandroid": "PlayStation",
    "com.epicgames.portal": "Epic Games",
    "com.max.xiaoheihe": "小黑盒/Xiaoheihe",
    # Other
    "com.wisentsoft.chinapost.android": "中国邮政/China Post",
    "com.fcbox.hiveconsumer": "丰巢/Hive Box",
    "com.cxincx.xxjz": "随手记/Suishouji",
    "com.tplink.ipc": "TP-Link Tapo",
    "io.heckel.ntfy": "ntfy",
    "com.sohu.inputmethod.sogouoem": "搜狗输入法/Sogou",
    "com.podcast.podcasts": "Podcasts",
    "com.jmchn.typhoon": "台风追踪/Typhoon",
    "com.netease.uuremote": "UU加速器/UU Booster",
    "cn.com.chsi.chsiapp": "学信网/CHSI",
    "com.incon.timetable": "课程表/Timetable",
    "cn.edu.hit.welink": "WeLink",
}

# Manual aliases that cannot be derived from display names. Copied from
# GUIClaw; MobileWorld synthetic *targets* are filtered out when building
# the ADB lookup table.
_ANDROID_APP_ALIASES_BASE: dict[str, str] = {
    "mail": "com.gmailclone",
    "mail app": "com.gmailclone",
    "email": "com.gmailclone",
    "email app": "com.gmailclone",
    "gmail": "com.gmailclone",
    "gmail app": "com.gmailclone",
    "gmailclone": "com.gmailclone",
    "gmail clone": "com.gmailclone",
    "google mail": "com.gmailclone",
    "google gmail": "com.gmailclone",
    "com.google.android.gm": "com.gmailclone",
    "messages": "com.google.android.apps.messaging",
    "messages app": "com.google.android.apps.messaging",
    "messaging": "com.google.android.apps.messaging",
    "messaging app": "com.google.android.apps.messaging",
    "sms": "com.google.android.apps.messaging",
    "sms app": "com.google.android.apps.messaging",
    "text message": "com.google.android.apps.messaging",
    "text messages": "com.google.android.apps.messaging",
    "com.android.mms": "com.google.android.apps.messaging",
    "com.android.messaging": "com.google.android.apps.messaging",
    "calendar": "org.fossify.calendar",
    "calendar app": "org.fossify.calendar",
    "fossify calendar": "org.fossify.calendar",
    "com.android.calendar": "org.fossify.calendar",
    "com.google.android.calendar": "org.fossify.calendar",
    "files": "com.google.android.documentsui",
    "files app": "com.google.android.documentsui",
    "file manager": "com.google.android.documentsui",
    "documents": "com.google.android.documentsui",
    "documentsui": "com.google.android.documentsui",
    "downloads": "com.google.android.documentsui",
    "downloads folder": "com.google.android.documentsui",
    "download folder": "com.google.android.documentsui",
    "download directory": "com.google.android.documentsui",
    "files-documents": "com.google.android.documentsui",
    "com.android.documentsui": "com.google.android.documentsui",
    "contacts": "com.google.android.contacts",
    "contacts app": "com.google.android.contacts",
    "phone contacts": "com.google.android.contacts",
    "com.android.contacts": "com.google.android.contacts",
    "phone": "com.google.android.dialer",
    "phone app": "com.google.android.dialer",
    "dialer": "com.google.android.dialer",
    "com.android.dialer": "com.google.android.dialer",
    "clock": "com.google.android.deskclock",
    "clock app": "com.google.android.deskclock",
    "deskclock": "com.google.android.deskclock",
    "desk clock": "com.google.android.deskclock",
    "google clock": "com.google.android.deskclock",
    "alarm": "com.google.android.deskclock",
    "alarm clock": "com.google.android.deskclock",
    "com.android.deskclock": "com.google.android.deskclock",
    "mattermost": "com.mattermost.rnbeta",
    "mattermost app": "com.mattermost.rnbeta",
    "mattermost beta": "com.mattermost.rnbeta",
    "mattermost rnbeta": "com.mattermost.rnbeta",
    "com.mattermost": "com.mattermost.rnbeta",
    "com.mattermost.rn": "com.mattermost.rnbeta",
    "slack": "com.mattermost.rnbeta",
    "mastodon": "org.joinmastodon.android.mastodon",
    "mastodon app": "org.joinmastodon.android.mastodon",
    "open talk": "org.joinmastodon.android.mastodon",
    "opentalk": "org.joinmastodon.android.mastodon",
    "org.joinmastodon.android": "org.joinmastodon.android.mastodon",
    "taodian": "com.testmall.app",
    "tao dian": "com.testmall.app",
    "淘店": "com.testmall.app",
    "taobao": "com.testmall.app",
    "com.taobao.taobao": "com.testmall.app",
    "gallery": "gallery.photomanager.picturegalleryapp.imagegallery",
    "gallery app": "gallery.photomanager.picturegalleryapp.imagegallery",
    "photo gallery": "gallery.photomanager.picturegalleryapp.imagegallery",
    "pictures": "gallery.photomanager.picturegalleryapp.imagegallery",
    "com.android.gallery3d": "gallery.photomanager.picturegalleryapp.imagegallery",
    "docreader": "at.tomtasche.reader",
    "doc reader": "at.tomtasche.reader",
    "document reader": "at.tomtasche.reader",
    "open document reader": "at.tomtasche.reader",
    "opendocument reader": "at.tomtasche.reader",
    "pdf reader": "at.tomtasche.reader",
    "reader": "at.tomtasche.reader",
    "reader app": "at.tomtasche.reader",
    "android settings": "com.android.settings",
    "system settings": "com.android.settings",
    "device settings": "com.android.settings",
    "phone settings": "com.android.settings",
    "com.google.android.settings.intelligence": "com.android.settings",
    "com.android.settings.intelligence": "com.android.settings",
    "google chrome": "com.android.chrome",
    "chrome browser": "com.android.chrome",
    "launcher": "com.google.android.apps.nexuslauncher",
    "nexuslauncher": "com.google.android.apps.nexuslauncher",
    "pixel launcher": "com.google.android.apps.nexuslauncher",
    "intentresolver": "com.android.intentresolver",
    "intent resolver": "com.android.intentresolver",
    "wechat": "com.tencent.mm",
    "weixin": "com.tencent.mm",
    "alipay": "com.eg.android.AlipayGphone",
    "zhifubao": "com.eg.android.AlipayGphone",
    "jd": "com.jingdong.app.mall",
    "jingdong": "com.jingdong.app.mall",
    "meituan": "com.sankuai.meituan",
    "douyin": "com.ss.android.ugc.aweme",
    "tiktok": "com.ss.android.ugc.aweme",
    "bilibili": "tv.danmaku.bili",
    "bilibili app": "tv.danmaku.bili",
    "bili": "tv.danmaku.bili",
    "b站": "tv.danmaku.bili",
    "b 站": "tv.danmaku.bili",
    "哔哩哔哩动画": "tv.danmaku.bili",
    "哔站": "tv.danmaku.bili",
    "小破站": "tv.danmaku.bili",
    "youtube app": "com.google.android.youtube",
    "yt": "com.google.android.youtube",
    "油管": "com.google.android.youtube",
    "ximalaya": "com.ximalaya.ting.android",
    "himalaya": "com.ximalaya.ting.android",
    "xmly": "com.ximalaya.ting.android",
    "喜马拉雅fm": "com.ximalaya.ting.android",
    "qqmusic": "com.tencent.qqmusic",
    "kugou": "com.kugou.android",
    "iqiyi": "com.qiyi.video",
    "kuaishou": "com.smile.gifmaker",
    "kwai": "com.smile.gifmaker",
    "toutiao": "com.ss.android.article.news",
    "didi": "com.sdu.didi.psnger",
    "weibo": "com.sina.weibo",
    "zhihu": "com.zhihu.android",
    "xhs": "com.xingin.xhs",
    "redbook": "com.xingin.xhs",
    "rednote": "com.xingin.xhs",
    "xiaohongshu": "com.xingin.xhs",
    "pinduoduo": "com.xunmeng.pinduoduo",
    "xianyu": "com.taobao.idlefish",
    "ctrip": "ctrip.android.view",
    "ctrip.com": "ctrip.android.view",
    "携程旅行": "ctrip.android.view",
    "lark": "com.ss.android.lark",
    "feishu": "com.ss.android.lark",
    "dingtalk": "com.alibaba.android.rimet",
    "wecom": "com.tencent.wework",
    "voov": "com.tencent.wemeet.app",
    "tencent meeting": "com.tencent.wemeet.app",
    "tencent docs": "com.tencent.docs",
    "weread": "com.tencent.weread",
    "fanqie": "com.dragon.read",
    "qidian": "com.qidian.QDReader",
    "baidu": "com.baidu.searchbox",
    "maps": "com.google.android.apps.maps",
    "maps app": "com.google.android.apps.maps",
    "map": "com.google.android.apps.maps",
    "amap": "com.autonavi.minimap",
    "gaode": "com.autonavi.minimap",
    "google map": "com.google.android.apps.maps",
    "google maps": "com.google.android.apps.maps",
    "baidu map": "com.baidu.BaiduMap",
    "baidu maps": "com.baidu.BaiduMap",
    "tencent map": "com.tencent.map",
    "tencent maps": "com.tencent.map",
    "play store": "com.android.vending",
    "google play": "com.android.vending",
    "twitter": "com.twitter.android",
}

_ANDROID_ADB_APP_ALIASES_BASE: dict[str, str] = {
    "mastodon": "org.joinmastodon.android",
    "mastodon app": "org.joinmastodon.android",
    "org.joinmastodon.android": "org.joinmastodon.android",
    "org.joinmastodon.android.mastodon": "org.joinmastodon.android",
}

_MOBILEWORLD_PACKAGES = frozenset({
    "com.gmailclone",
    "org.fossify.calendar",
    "com.mattermost.rnbeta",
    "org.joinmastodon.android.mastodon",
    "com.testmall.app",
    "gallery.photomanager.picturegalleryapp.imagegallery",
    "at.tomtasche.reader",
})

_ANDROID_ALIAS_SUFFIXES = (
    " app",
    " application",
    " 应用",
    " 软件",
    " 客户端",
    "app",
    "应用",
    "软件",
    "客户端",
)
_ANDROID_ALIAS_PREFIXES = (
    "打开",
    "启动",
    "开启",
    "运行",
    "进入",
    "open ",
    "launch ",
    "start ",
)


def _android_display_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for package, display in _ANDROID_PACKAGE_DISPLAY_NAMES.items():
        for part in display.split("/"):
            key = part.strip().lower()
            if key and key not in aliases:
                aliases[key] = package
        full = display.strip().lower()
        if full not in aliases:
            aliases[full] = package
    return aliases


def _build_android_adb_aliases() -> dict[str, str]:
    aliases = _android_display_aliases()
    for alias, package in _ANDROID_APP_ALIASES_BASE.items():
        if package in _MOBILEWORLD_PACKAGES:
            continue
        if _ANDROID_PACKAGE_RE.match(alias):
            continue
        aliases[alias] = package
    aliases.update(_ANDROID_ADB_APP_ALIASES_BASE)
    return aliases


_ANDROID_ADB_APP_ALIASES = _build_android_adb_aliases()


def _lookup_android_alias_from(lowered: str, aliases: Mapping[str, str]) -> str | None:
    candidates = [lowered]
    for prefix in _ANDROID_ALIAS_PREFIXES:
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            candidates.append(lowered[len(prefix) :].strip())
    for candidate in candidates:
        package = aliases.get(candidate)
        if package:
            return package
        for suffix in _ANDROID_ALIAS_SUFFIXES:
            if candidate.endswith(suffix) and len(candidate) > len(suffix):
                stem = candidate[: -len(suffix)].strip()
                if stem:
                    package = aliases.get(stem)
                    if package:
                        return package
    return None


def _merge_extra_aliases(extra_aliases: Mapping[str, str] | None) -> dict[str, str]:
    if not extra_aliases:
        return _ANDROID_ADB_APP_ALIASES
    aliases = dict(_ANDROID_ADB_APP_ALIASES)
    for raw_key, raw_package in extra_aliases.items():
        key = " ".join(str(raw_key).strip().lower().split())
        package = str(raw_package).strip()
        if key and package:
            aliases[key] = package
    return aliases


def normalize_adb_app_identifier(
    app: str,
    extra_aliases: Mapping[str, str] | None = None,
) -> str:
    """Resolve a human app name to an Android package for ADB commands.

    Already-qualified package names are preserved (including case). Names
    that cannot be resolved return ``"unknown"``.
    """
    cleaned = " ".join((app or "").strip().strip("\"'").split())
    if not cleaned:
        return "unknown"

    aliases = _merge_extra_aliases(extra_aliases)
    lowered = cleaned.lower()
    package = _lookup_android_alias_from(lowered, aliases)
    if package:
        return package
    if _ANDROID_PACKAGE_RE.match(cleaned):
        return cleaned
    return "unknown"


def is_android_package_name(value: str) -> bool:
    return bool(_ANDROID_PACKAGE_RE.match(value or ""))
