import express from "express";
import cors from "cors";
import Report from "./models/reports.js";
import multer from "multer";
import mongoose from "mongoose";
import axios from "axios";

const app = express();
app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.send("Backend is running ✅");
});

mongoose
  .connect(
    "mongodb+srv://vishwasrudramurthy26_db_user:0E7jVElYSAjbKnX5@cluster0.v4yfwoz.mongodb.net/civix-db?retryWrites=true&w=majority&appName=Cluster0"
  )
  .then(() => console.log("✅ MongoDB Connected"))
  .catch((err) => console.log(err));

app.listen(8000, () => {
  console.log("Server running on http://localhost:8000");
});

//
const upload = multer({ dest: "uploads/" }); // store uploaded images

// Create report (called after AI prediction)
app.post("/report", upload.single("image"), async (req, res) => {
  const { name, age, email, conditions, result } = req.body;

  const report = await Report.create({
    name,
    age,
    email,
    conditions,
    result,
    imagePath: req.file?.path || null,
  });

  res.json({ success: true, report });
});

// Get all reports
app.get("/reports", async (req, res) => {
  const reports = await Report.find().sort({ createdAt: -1 });
  res.json(reports);
});

//

app.post("/analyze", upload.single("image"), async (req, res) => {
  const { name, age, email, conditions } = req.body;

  // Send image to FastAPI Model
  const response = await axios.post("http://localhost:9000/predict", req.file, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  const result = response.data.result; // This is class ID

  // Save report
  const report = await Report.create({
    name,
    age,
    email,
    conditions,
    result,
    imagePath: req.file.path,
  });

  res.json(report);
});
///
