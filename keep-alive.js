// keep-alive.js
const axios = require("axios");
setInterval(async () => {
  try {
    await axios.get("https://vector-bot-backend-rkgf.onrender.com/health");
    console.log("Keep-alive ping successful");
  } catch (error) {
    console.error("Keep-alive ping failed:", error.message);
  }
}, 5 * 60 * 1000); // Every 5 minutes
