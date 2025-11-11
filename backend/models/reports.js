import mongoose from "mongoose";

const ReportSchema = new mongoose.Schema({
  name: String,
  age: Number,
  email: String,
  conditions: String,
  result: String, // Example: "No DR", "Moderate DR", etc.
  imagePath: String, // File path for stored image
  createdAt: { type: Date, default: Date.now },
});

export default mongoose.model("Report", ReportSchema);
