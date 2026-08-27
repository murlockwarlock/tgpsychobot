const PM2_LOG_DIR = `${process.env.PM2_HOME || "/root/.pm2"}/logs`;

module.exports = {
  apps: [
    {
      name: "tg_kontentzavod322bot_fulltest",
      script: "/root/telegram_bots/newbots/main.py",
      error_file: `${PM2_LOG_DIR}/psy5d_fulltest-error.log`,
      out_file: `${PM2_LOG_DIR}/psy5d_fulltest-out.log`,
      cwd: "/root/telegram_bots/newbots",
      interpreter: "/root/telegram_bots/venv/bin/python",
      env: {
        BOT_TOKEN: process.env.PSY5D_FULLTEST_BOT_TOKEN,
        DATABASE_URL: process.env.PSY5D_FULLTEST_DATABASE_URL,
        OWNER_IDS: process.env.PSY5D_FULLTEST_OWNER_IDS || "806750628",
        APP_PORT: 8100,
        WEBHOOK_PATH_PREFIX: "/psy5d_fulltest",
        TELEGRAM_DELIVERY_MODE: "polling",
        SERVER_IP: process.env.SERVER_IP || "127.0.0.1",
        BASE_WEBHOOK_URL: process.env.BASE_WEBHOOK_URL || "http://localhost",
        GEMINI_PROXY: process.env.GEMINI_PROXY,
        NO_PROXY: process.env.NO_PROXY || "localhost,127.0.0.1",
      },
    },
  ],
};
