package com.example.leafimageretrievalandrecognitionapplication;

import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Base64;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.QuerySnapshot;

import org.json.JSONArray;
import org.json.JSONException;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class FinalResultActivity extends AppCompatActivity {

    private RecyclerView recyclerView;
    private ResultAdapter resultAdapter;

    private TextView noSimilarImagesTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_final_result);

        // Retrieve the response data from the intent
        Intent intent = getIntent();
        if (intent != null) {
            String indicesString = intent.getStringExtra("indices");
            String resultString = intent.getStringExtra("result");

            try {
                JSONArray indicesArray = new JSONArray(indicesString);
                JSONArray resultArray = new JSONArray(resultString);

                // Convert the JSONArray objects to List<Integer> and List<Double> respectively
                List<Integer> indices = new ArrayList<>();
                List<String> results = new ArrayList<>();

                for (int i = 0; i < indicesArray.length(); i++) {
                    int index = indicesArray.getInt(i);
                    String result = resultArray.getString(i);
                    indices.add(index);
                    results.add(result);
                }

                // Display progress dialog
                progressDialog = new ProgressDialog(FinalResultActivity.this);
                progressDialog.setMessage("Loading Result...");
                progressDialog.setCancelable(false);
                progressDialog.show();

                // Initialize the RecyclerView
                recyclerView = findViewById(R.id.recyclerView);
                recyclerView.setLayoutManager(new LinearLayoutManager(this));

                // Fetch data from Firestore for each index and add it to the adapter
                List<String> speciesList = new ArrayList<>();
                List<String> descriptionList = new ArrayList<>();
                List<Bitmap> imageList = new ArrayList<>();

                // Create a counter to keep track of the number of fetched data
                AtomicInteger counter = new AtomicInteger(0);

                // Initialize the adapter
                resultAdapter = new ResultAdapter(speciesList, descriptionList, imageList, results);

                // Set the adapter to the RecyclerView
                recyclerView.setAdapter(resultAdapter);

                // Iterate over the indices and fetch data from Firestore
                for (int index : indices) {
                    getDataFromFirestore(index, speciesList, descriptionList, imageList, counter, results.size());
                }

                // Check if the indices list is empty and show the "No Similar Images" message
                noSimilarImagesTextView = findViewById(R.id.noSimilarImagesTextView);
                if (indices.isEmpty()) {
                    noSimilarImagesTextView.setVisibility(View.VISIBLE);
                    progressDialog.dismiss();
                } else {
                    noSimilarImagesTextView.setVisibility(View.GONE);
                }

            } catch (JSONException e) {
                e.printStackTrace();
            }
        }

        ImageView backIcon = findViewById(R.id.backIcon);
        backIcon.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Intent intent = new Intent(FinalResultActivity.this, SelectedPhotoActivity.class);
                startActivity(intent);
                finish();
            }
        });
    }

    private ProgressDialog progressDialog;

    // Inside centerProgressDialog method
    private void centerProgressDialog() {
        if (progressDialog.getWindow() != null) {
            WindowManager.LayoutParams layoutParams = progressDialog.getWindow().getAttributes();
            layoutParams.gravity = Gravity.CENTER;
            progressDialog.getWindow().setAttributes(layoutParams);
        }
    }

    private Bitmap decodeBase64Image(String imageBase64) {
        try {
            byte[] decodedBytes = Base64.decode(imageBase64, Base64.DEFAULT);
            return BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.length);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private Bitmap resizeBitmap(Bitmap bitmap, int desiredWidth, int desiredHeight) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // Calculate the scale factor to resize the image
        float scaleWidth = (float) desiredWidth / width;
        float scaleHeight = (float) desiredHeight / height;

        // Create a matrix for the resizing
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);

        // Resize the Bitmap using the matrix and return the resized Bitmap
        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, false);
    }

    private void getDataFromFirestore(int index, List<String> speciesList, List<String> descriptionList, List<Bitmap> imageList, AtomicInteger counter, int totalResults) {
        FirebaseFirestore db = FirebaseFirestore.getInstance();
        db.collection("LeafDatabase")
                .whereEqualTo("Index", index)
                .get()
                .addOnCompleteListener(new OnCompleteListener<QuerySnapshot>() {
                    @Override
                    public void onComplete(@NonNull Task<QuerySnapshot> task) {
                        if (task.isSuccessful()) {
                            List<DocumentSnapshot> documents = task.getResult().getDocuments();
                            if (!documents.isEmpty()) {
                                DocumentSnapshot document = documents.get(0);
                                String species = document.getString("Species");
                                String description = document.getString("Description");
                                String imageBase64 = document.getString("Image");

                                // Convert the Base64 image string to a Bitmap
                                Bitmap imageBitmap = decodeBase64Image(imageBase64);

                                // Add the fetched data to the respective lists
                                speciesList.add(species);
                                descriptionList.add(description);
                                imageList.add(imageBitmap);

                                // Increment the counter
                                int count = counter.incrementAndGet();

                                // Notify the adapter that data has changed
                                resultAdapter.notifyDataSetChanged();

                                // Check if all data has been fetched
                                if (count == totalResults) {
                                    progressDialog.dismiss();
                                }
                            }
                        }
                    }
                });
    }

    private class ResultAdapter extends RecyclerView.Adapter<ResultAdapter.ResultViewHolder> {

        private List<String> speciesList;
        private List<String> descriptionList;
        private List<Bitmap> imageList;
        private List<String> results; // List to hold the scores

        public ResultAdapter(List<String> speciesList, List<String> descriptionList, List<Bitmap> imageList, List<String> results) {
            this.speciesList = speciesList;
            this.descriptionList = descriptionList;
            this.imageList = imageList;
            this.results = results;
        }

        @Override
        public ResultViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View itemView = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_layout, parent, false);
            return new ResultViewHolder(itemView);
        }

        @Override
        public void onBindViewHolder(@NonNull ResultViewHolder holder, int position) {
            String species = speciesList.get(position);
            Bitmap imageBitmap = imageList.get(position);
            String score = results.get(position);

            holder.labelNumberTextView.setText(String.valueOf(position + 1));
            holder.speciesTextView.setText(species);
            holder.resultsTextView.setText(score); // Display the score

            // Resize the imageBitmap for the RecyclerView
            Bitmap resizedImageBitmap = resizeBitmap(imageBitmap, 80, 80);
            holder.imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
            holder.imageView.setImageBitmap(resizedImageBitmap);

            holder.viewDetailButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    int index = holder.getAdapterPosition();
                    String species = speciesList.get(index);
                    String description = descriptionList.get(index);
                    Bitmap image = imageList.get(index);

                    // Set the image using ImageData class
                    ImageData.setImage(image);

                    Intent intent = new Intent(v.getContext(), LeafDetailActivity.class);
                    intent.putExtra("species", species);
                    intent.putExtra("description", description);
                    v.getContext().startActivity(intent);
                }
            });
        }

        @Override
        public int getItemCount() {
            return speciesList.size();
        }

        public class ResultViewHolder extends RecyclerView.ViewHolder {
            TextView labelNumberTextView;
            ImageView imageView;
            TextView speciesTextView;
            TextView resultsTextView; // TextView to display the score
            Button viewDetailButton;

            public ResultViewHolder(@NonNull View itemView) {
                super(itemView);
                labelNumberTextView = itemView.findViewById(R.id.labelNumberTextView);
                imageView = itemView.findViewById(R.id.imageView);
                speciesTextView = itemView.findViewById(R.id.speciesTextView);
                resultsTextView = itemView.findViewById(R.id.resultsTextView); // Initialize the scoreTextView
                viewDetailButton = itemView.findViewById(R.id.viewDetailButton);

                // Set the image view scale type to fit center
                imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
            }
        }
    }
}
