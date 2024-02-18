const express = require("express");
const fs = require("fs");
const path = require("path");
const bodyParser = require("body-parser");

// Assuming you have a JavaScript version of DoctorRecommendationModel and config
const DoctorRecommendationModel = require("../models/DoctorRecommendationModel");
const DB_CONFIG = require("./config");

const app = express();
const port = 3000; // Default port or you can choose

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// Load your model
const modelPath = path.join(
  __dirname,
  "..",
  "models",
  "doctor_recommendation_model.pkl"
);
// This part needs to be adapted to JavaScript, as loading a Python pickle file directly in JS is not straightforward
const model = new DoctorRecommendationModel();
model.loadModel(modelPath);

app.post("/recommend-doctors", (req, res) => {
  const data = req.body;
  const patientIndex = data.patient_index;
  const topN = data.top_n || 4; // Default to 4 recommendations if not specified

  // Get recommendations
  // This needs to be adapted to how your model is implemented in JS
  model
    .recommendDoctors(patientIndex, topN)
    .then((recommendedDoctors) => {
      // Assuming recommendDoctors method returns a Promise that resolves to the recommended doctors
      // Convert recommendations to a list of dicts (or similar, depending on your data structure)
      const recommendations = recommendedDoctors.map((doc) => doc.toDict());
      res.json(recommendations);
    })
    .catch((error) => {
      console.error("Error recommending doctors:", error);
      res.status(500).send("Error processing your request");
    });
});

app.listen(port, () => {
  console.log(
    `Doctor recommendation app listening at http://localhost:${port}`
  );
});
