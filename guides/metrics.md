```c++
int main()
{ cout<<"Hello\n";
}
```

Наименование | Количество
------------ | ---------:
Апельсины    | 5
Яблоки       | 120
Груши        | 25

```width:250px|uml
hide circle
Object <|-- ArrayList

Object : equals()
ArrayList : Object[] elementData
ArrayList : size()
```