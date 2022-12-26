

# Introduction
In the following coolab notebook we will perform three examples:
 *  A descent gradient exercise.
 *  One with support vector machine (SVM)
 *   One with the SVM kernel
 
# Descent gradient exercise
In the following example we will use our own input and output data. These were created as follows:
```python
#We create our data set
#Imput data
x = np.array([49,69,89,99,109])

#Ouput data
y = np.array([124,95,71,45,18])

```
Where we will make the gradient for the following function step by step:

**Target function:** $MSE(\theta) = \dfrac{1}{m} \sum (Y - \theta X)^2$

**Gradient:** $\nabla MSE(\theta) = \dfrac{2}{m} \sum (\theta X-Y)X$

**Updating equation:**

$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$

$x_{k} = x_{k-1} - \alpha_{k-1} \nabla f(x_{k-1})$

**We want to find the optimal values of:** $\theta$

$$x_{k} = \begin{bmatrix}\theta_0\\ \theta_1\end{bmatrix}$$

**Replacing in the update equation:**

$$\begin{bmatrix} \theta_{0_{k}}\\ \theta_{1_k}\end{bmatrix}= \begin{bmatrix} \theta_{0_{k-1}}\\ \theta_{1_{n-1}} \end{bmatrix}- \alpha \nabla MSE(\begin{bmatrix}\theta_{0_{k-1}}\\ \theta_{1_{k-1}}\end{bmatrix})$$


**Replacing the gradient:**

$$\begin{bmatrix}\theta_{0_k}\\ \theta_{1_k}\end{bmatrix}= \begin{bmatrix}\theta_{0_{k-1}}\\ \theta_{1_{k-1}}\end{bmatrix}- \alpha\frac{2}{m}x^T([\theta_1\cdot x + \theta_0]-y)$$

In order to continue with the step-by-step exercise we perform the following function to calculate the **mse** step by step
```python
#Define a mse function
def mse(y, y_hat):
  mse_value = 1/y.size * sum((y - y_hat) ** 2)
  return mse_value
```

Once the **mse** has been calculated, we obtain the following plan:
![image](https://user-images.githubusercontent.com/115313115/209488231-e63ee037-a7ab-49f3-8027-df8b8ba28f13.png)

# Linear SVM

For this example we will use the **pima-indians-diabetes.csv** dataset, in order to predict the output using the SVM method, in this case we will normalize our outputs and separate our dataset manually into validation and test data, finally we will compare the test outputs, with the predicted outputs with the test data. Some code extracts to take into account are:
```python

#Build dataset
dataset = df.iloc[:,:-1].values
target = df.iloc[:,8:].values

#Normalize x's
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataset_norm = scaler.fit_transform(dataset)

#Split manually the inputs and outputs
x_train = dataset_norm[0:536,0:-1]
x_test = dataset_norm[536:,0:-1]

y_train = target[0:536]
y_test = target[536:]

#Build our model
from sklearn.svm import LinearSVC

svc = LinearSVC(C=100, loss="hinge")

svc.fit(x_train, y_train)
```
# Kernel SVM

For this example we will obtain our data from the sklearn **make circles** library, once we obtain our data we will normalize the inputs and do the separation of validation and test data manually, we will prepare the model and train it, we will evaluate the results with the confusion matrix, the precision and recall, obtained.
Some code excerpts relevant to this exercise:
```python
#Create our data
X, y = make_moons(n_samples=100, noise=0.15, random_state=0)

#Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataset_norm = scaler.fit_transform(X)

#Split data
x_train = dataset_norm[0:70,:]
x_test = dataset_norm[70:,:]

y_train = y[0:70]
y_test = y[70:]

#Build our model
from sklearn.svm import SVC

svk = SVC(kernel='poly', degree=3, coef0=1, C=5)
svk.fit(x_train, y_train)
```
### Where our model results in the following graph:

![image](https://user-images.githubusercontent.com/115313115/209490217-0bc50912-62c9-48cf-b252-2908db08f628.png)

As we saw in this case we use the **kernel trick** in order to implement a classifier that separates the points different from a straight line, another way to use linear classifiers for nonlinearly separable problems is to add more variables from the existing ones but increasing their degree.More variables from the existing ones but increasing their degree. This would look like this:

![image](https://user-images.githubusercontent.com/115313115/209490550-95806d8c-1152-4b85-8205-e3e01e8834b2.png)





