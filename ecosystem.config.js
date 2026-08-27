const PM2_LOG_DIR = `${process.env.PM2_HOME || "/root/.pm2"}/logs`;
const GEMINI_PROXY = process.env.GEMINI_PROXY || "";
const DEEPSEEK_BASE_URL = process.env.DEEPSEEK_BASE_URL || "";
const DEEPSEEK_PROXY = process.env.DEEPSEEK_PROXY || "";
const TELEGRAM_PROXY = process.env.TELEGRAM_PROXY || "";
const HTTPS_PROXY = process.env.HTTPS_PROXY || "";
const HTTP_PROXY = process.env.HTTP_PROXY || "";
const NO_PROXY = process.env.NO_PROXY || "localhost,127.0.0.1,::1";

const apps = [
  {
    name: "tg_autobusbusbot_new",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/psy5d_new-error.log`,
    out_file: `${PM2_LOG_DIR}/psy5d_new-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_AUTOBUSBUSBOT_TOKEN || "",
      "OWNER_IDS": "272982544,979514796,806750628",
      "DATABASE_URL": process.env.TELEGRAM_AUTOBUSBUSBOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8081,
      "WEBHOOK_PATH_PREFIX": "/bot1",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_alena322bot_new",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/veraveda_new-error.log`,
    out_file: `${PM2_LOG_DIR}/veraveda_new-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_ALENA322BOT_TOKEN || "",
      "OWNER_IDS": "272982544,979514796,806750628",
      "DATABASE_URL": process.env.TELEGRAM_ALENA322BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8082,
      "WEBHOOK_PATH_PREFIX": "/bot2",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_someonewithyou01_bot_new",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/someone01_new-error.log`,
    out_file: `${PM2_LOG_DIR}/someone01_new-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONEWITHYOU01_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,979514796,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONEWITHYOU01_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8083,
      "WEBHOOK_PATH_PREFIX": "/bot3",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_someonewithyou02_bot_new",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/someone02_new-error.log`,
    out_file: `${PM2_LOG_DIR}/someone02_new-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONEWITHYOU02_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONEWITHYOU02_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8084,
      "WEBHOOK_PATH_PREFIX": "/bot4",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_veraveda777_bot_legacy",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/veraveda_legacy-error.log`,
    out_file: `${PM2_LOG_DIR}/veraveda_legacy-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_VERAVEDA777_BOT_TOKEN || "",
      "OWNER_IDS": "979514796,272982544,806750628",
      "DATABASE_URL": process.env.TELEGRAM_VERAVEDA777_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8085,
      "WEBHOOK_PATH_PREFIX": "/bot_legacy_1",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling",
    }
  },
  {
    name: "tg_someonelikeyouai03_bot_legacy",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/someone02_legacy-error.log`,
    out_file: `${PM2_LOG_DIR}/someone02_legacy-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONELIKEYOUAI03_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONELIKEYOUAI03_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8086,
      "WEBHOOK_PATH_PREFIX": "/bot_legacy_2",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_someonelikeyouai04_bot_legacy",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/someone01_legacy-error.log`,
    out_file: `${PM2_LOG_DIR}/someone01_legacy-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONELIKEYOUAI04_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONELIKEYOUAI04_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8087,
      "WEBHOOK_PATH_PREFIX": "/bot_legacy_3",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_psy5d_bot_legacy",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/psy5d_legacy-error.log`,
    out_file: `${PM2_LOG_DIR}/psy5d_legacy-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_PSY5D_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,806750628",
      "DATABASE_URL": process.env.TELEGRAM_PSY5D_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8088,
      "WEBHOOK_PATH_PREFIX": "/bot_legacy_4",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_someonelikeyouai_bot_legacy",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/test01_legacy-error.log`,
    out_file: `${PM2_LOG_DIR}/test01_legacy-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONELIKEYOUAI_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONELIKEYOUAI_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8089,
      "WEBHOOK_PATH_PREFIX": "/bot_legacy_5",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_someonelikeyouai02_bot_legacy",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/test02_legacy-error.log`,
    out_file: `${PM2_LOG_DIR}/test02_legacy-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONELIKEYOUAI02_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONELIKEYOUAI02_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8090,
      "WEBHOOK_PATH_PREFIX": "/bot_legacy_6",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_someonewithyou03_bot_new",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/someone03_new-error.log`,
    out_file: `${PM2_LOG_DIR}/someone03_new-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONEWITHYOU03_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,979514796,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONEWITHYOU03_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8091,
      "WEBHOOK_PATH_PREFIX": "/bot5",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_someonelikeyouai05_bot_new",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/someone04_new-error.log`,
    out_file: `${PM2_LOG_DIR}/someone04_new-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONELIKEYOUAI05_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,979514796,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONELIKEYOUAI05_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8092,
      "WEBHOOK_PATH_PREFIX": "/bot6",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_someonewithyou05_bot_new",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/someone05_new-error.log`,
    out_file: `${PM2_LOG_DIR}/someone05_new-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONEWITHYOU05_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,979514796,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONEWITHYOU05_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8093,
      "WEBHOOK_PATH_PREFIX": "/bot7",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_someonewithyou06_bot_new",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/someone06_new-error.log`,
    out_file: `${PM2_LOG_DIR}/someone06_new-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONEWITHYOU06_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,979514796,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONEWITHYOU06_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8094,
      "WEBHOOK_PATH_PREFIX": "/bot8",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_someonewithyou04_bot_new",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/someone07_new-error.log`,
    out_file: `${PM2_LOG_DIR}/someone07_new-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_SOMEONEWITHYOU04_BOT_TOKEN || "",
      "OWNER_IDS": "272982544,979514796,806750628",
      "DATABASE_URL": process.env.TELEGRAM_SOMEONEWITHYOU04_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8095,
      "WEBHOOK_PATH_PREFIX": "/bot9",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling"
    }
  },
  {
    name: "tg_yourself_way_bot_new",
    script: "/root/telegram_bots/newbots/main.py",
    error_file: `${PM2_LOG_DIR}/yourself_way_bot-error.log`,
    out_file: `${PM2_LOG_DIR}/yourself_way_bot-out.log`,
    interpreter: "/root/telegram_bots/venv/bin/python",
    env: {
      "BOT_TOKEN": process.env.TELEGRAM_YOURSELF_WAY_BOT_TOKEN || "",
      "OWNER_IDS": "979514796,272982544,806750628,1372041348,100263465646,100005511792,100178646155",
      "DATABASE_URL": process.env.TELEGRAM_YOURSELF_WAY_BOT_DATABASE_URL || "",
      "SERVER_IP": "77.91.112.200",
      "APP_PORT": 8098,
      "WEBHOOK_PATH_PREFIX": "/bot10",
      "BASE_WEBHOOK_URL": "https://bots.psysoldatov.ru:8443",
      "TELEGRAM_DELIVERY_MODE": "polling",
    }
  },
  {
    name: "max_se13639182_bot_legacy",
    script: "/root/telegram_bots/venv/bin/python",
    error_file: `${PM2_LOG_DIR}/max_veraveda_legacy-error.log`,
    out_file: `${PM2_LOG_DIR}/max_veraveda_legacy-out.log`,
    args: "-m max_messenger_bot.app",
    interpreter: "none",
    cwd: "/root/telegram_bots/newbots",
    env: {
      "MAX_BOT_TOKEN": process.env.MAX_SE13639182_BOT_TOKEN || "",
      "DATABASE_URL": process.env.MAX_SE13639182_BOT_DATABASE_URL || "",
      "OWNER_IDS": "100263465646,100005511792,100178646155",
      "MAX_USE_POLLING": "1",
      "MAX_BOT_NAME": "se13639182_bot",
      "MAX_APP_PORT": "8099",
      "MAX_LOG_LEVEL": "INFO"
    }
  },
  {
    name: "max_id519010411655_bot_new",
    script: "/root/telegram_bots/venv/bin/python",
    error_file: `${PM2_LOG_DIR}/max_yourself_way-error.log`,
    out_file: `${PM2_LOG_DIR}/max_yourself_way-out.log`,
    args: "-m max_messenger_bot.app",
    interpreter: "none",
    cwd: "/root/telegram_bots/newbots",
    env: {
      "MAX_BOT_TOKEN": process.env.MAX_ID519010411655_BOT_TOKEN || "",
      "DATABASE_URL": process.env.MAX_ID519010411655_BOT_DATABASE_URL || "",
      "OWNER_IDS": "100263465646,100005511792,100178646155",
      "MAX_USE_POLLING": "1",
      "MAX_BOT_NAME": "id519010411655_bot",
      "MAX_APP_PORT": "8097",
      "MAX_LOG_LEVEL": "INFO"
    }
  }
];

apps.forEach(app => {
  if (GEMINI_PROXY) {
    app.env.GEMINI_PROXY = GEMINI_PROXY;
  }
  if (DEEPSEEK_BASE_URL) {
    app.env.DEEPSEEK_BASE_URL = DEEPSEEK_BASE_URL;
  }
  if (DEEPSEEK_PROXY) {
    app.env.DEEPSEEK_PROXY = DEEPSEEK_PROXY;
  }
  if (TELEGRAM_PROXY) {
    app.env.TELEGRAM_PROXY = TELEGRAM_PROXY;
  }
  if (HTTPS_PROXY) {
    app.env.HTTPS_PROXY = HTTPS_PROXY;
  }
  if (HTTP_PROXY) {
    app.env.HTTP_PROXY = HTTP_PROXY;
  }
  app.env.NO_PROXY = NO_PROXY;
});

module.exports = {
  apps: apps
};
